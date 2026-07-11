// Package kt128 implements KT128 (KangarooTwelve) as specified in RFC 9861.
//
// KT128 is a tree-hash eXtendable-Output Function (XOF) built on TurboSHAKE128.
// When the input (the message plus the customization string and its length
// encoding) exceeds one 8192-byte chunk, it splits the input into chunks and
// computes a leaf chain value from each. On amd64 and arm64 the leaves are
// computed in parallel using SIMD-accelerated Keccak permutations; other
// targets and the purego build use a scalar fallback.
package kt128

import (
	"crypto/subtle"
	"hash"
	"math/bits"
	"slices"
)

const (
	// BlockSize is the KT128 chunk size in bytes.
	BlockSize = 8192

	leafDS   = 0x0B
	treeDS   = 0x06
	singleDS = 0x07

	// Hasher lifecycle states.
	stateSingle    uint8 = 0 // absorbing, single-node (< 1 chunk seen)
	stateTree      uint8 = 1 // absorbing, tree mode (S_0 flushed)
	stateFinalized uint8 = 2 // finalized and squeezable
)

// Hasher is an incremental KT128 instance.
type Hasher struct {
	buf        []byte       // buffered leaf data (tree mode only)
	c          []byte       // owned copy of the customization string
	final      sponge       // final-node sponge state
	pending    pendingState // partially-absorbed trailing leaf from a fused first write
	pos        uint64       // total bytes written via Write
	leafCount  uint64       // total leaf CVs written to final so far
	pendingLen int          // bytes absorbed into pending; 0 = no pending leaf
	state      uint8        // lifecycle: stateSingle -> stateTree -> stateFinalized
	ds         byte         // KT128 customization byte for finalization (singleDS or treeDS)
}

// New returns a new Hasher with the given customization string. The Hasher
// retains a copy of c, so later mutations of the caller's slice do not affect
// the output. Pass nil for no customization.
func New(c []byte) *Hasher {
	return &Hasher{c: slices.Clone(c)}
}

// BlockSize returns the KT128 chunk size in bytes (8192). Writes need not be a
// multiple of this size.
func (h *Hasher) BlockSize() int {
	return BlockSize
}

// Pos returns the total number of bytes written via Write.
func (h *Hasher) Pos() uint64 {
	return h.pos
}

// Write absorbs message bytes. It panics if called after Read, which finalizes
// the hasher.
func (h *Hasher) Write(p []byte) (int, error) {
	if h.state == stateFinalized {
		panic("kt128: Hasher is finalized")
	}

	n := len(p)
	fusedS0 := false

	if h.state == stateSingle {
		// Fused fast path: with S_0 and at least one full leaf contiguous in
		// p and nothing absorbed yet, process them together in one fused
		// kernel pass. Each arch decides how many chunks to take (and when
		// fusion would strand leaves in the buffer) via fuseS0Chunks. When
		// the pass consumes every complete chunk, the trailing partial
		// chunk's whole rate-blocks may ride an idle lane (fuseS0TailBlocks),
		// leaving a pending partially-absorbed leaf.
		nFuse, nTail := 0, 0
		if h.pos == 0 {
			chunks, tail := len(p)/BlockSize, len(p)%BlockSize
			nFuse = fuseS0Chunks(chunks, tail)
			if nFuse == chunks {
				nTail = fuseS0TailBlocks(chunks, tail)
			}
		}
		if nFuse >= 2 && h.startTreeModeFused(p, nFuse, nTail) {
			// The rest of p is ordinary leaf data — or, with a pending
			// leaf, its ragged remnant, buffered by extendPending below.
			p = p[nFuse*BlockSize+h.pendingLen:]
			fusedS0 = true
		} else {
			// Single-node finalization and tree-mode S_0 absorb the first
			// chunk into the final node identically, so message bytes are
			// absorbed eagerly with no buffering; the two modes diverge only
			// once the input exceeds one chunk.
			room := BlockSize - int(h.pos)
			if len(p) <= room {
				h.final.absorb(p)
				h.pos += uint64(n)
				return n, nil
			}
			h.final.absorb(p[:room])
			p = p[room:]
			h.startTreeMode()
		}
	}

	h.pos += uint64(n)

	// A pending trailing leaf holds the stream position: p continues it.
	if h.pendingLen > 0 {
		if p = h.extendPending(p); len(p) == 0 {
			return n, nil
		}
	}

	// A fused first write's chunk-aligned tail drains in place as well: the
	// fused pass makes this bulk traffic ending on a chunk boundary — the
	// same shape the direct path drains in place — but the leftover may sit
	// below the flush-unit gate, and streaming it costs up to seven chunks of
	// memcpy plus an allocation for the same narrow pass at finalization
	// (+11..38% at 80..120 KiB one-shots, Emerald Rapids). A ragged tail
	// still buffers whole so its trailing chunks can ride a fused pass with
	// the partial at finalization.
	if r := len(p) / BlockSize; fusedS0 && r >= 2 && len(p) == r*BlockSize {
		h.processLeafBatch(p, r)
		return n, nil
	}

	lanes := streamChunks
	flush := flushChunks()

	// Direct fast path: process chunks in place from p to avoid copying. With
	// buffered data present, mid-size writes keep the buffer-and-batch route
	// below so buffered chunks aren't pushed through narrow kernels
	// prematurely; where flushChunks() == streamChunks this reduces to the
	// flush-unit threshold alone.
	if len(p) >= flush*BlockSize && (len(h.buf) == 0 || len(p) >= lanes*BlockSize) {
		// Drain any buffered data: complete the partial tail with bytes from
		// p, then flush all buffered chunks as a single batch.
		if len(h.buf) > 0 {
			if partial := len(h.buf) % BlockSize; partial != 0 {
				need := BlockSize - partial
				h.bufferTail(p[:need])
				p = p[need:]
			}

			// Complete a SIMD-width batch from buffered whole chunks before
			// draining it. The bounded copy avoids sending a small buffered
			// remainder through a narrow kernel when this write can fill all
			// lanes.
			if buffered := len(h.buf) / BlockSize; buffered < lanes && len(p) >= (lanes-buffered)*BlockSize {
				take := (lanes - buffered) * BlockSize
				h.bufferTail(p[:take])
				p = p[take:]
			}
			h.processLeafBatch(h.buf, len(h.buf)/BlockSize)
			h.buf = h.buf[:0]
		}

		// Flush the architecture-selected complete-chunk prefix in place.
		// Complete message leaves need no lookahead, since the customization
		// suffix added at finalization is always non-empty.
		processable := len(p) / BlockSize
		nFlush := directFlushChunks(processable)
		if nFlush > 0 {
			h.processLeafBatch(p[:nFlush*BlockSize], nFlush)
			p = p[nFlush*BlockSize:]
		}

		// A chunk-aligned remainder of two or more after a whole-unit flush
		// drains in place as well: with no partial tail to extend, finalization
		// would push these same chunks through the same narrow kernels from the
		// buffer, so buffering buys no better pass and costs the copy plus its
		// allocation (up to seven chunks). A sub-chunk continuation could have
		// completed the buffered chunks into a whole batch, but a stream
		// overwhelmingly ends at a bulk write, and the miss costs one narrow
		// pass. A single trailing chunk still buffers — an x1 pass now or at
		// finalization costs the same, and a later write may pair it — and a
		// ragged tail still buffers whole, so its trailing chunks can ride a
		// fused pass with the partial at finalization.
		if r := len(p) / BlockSize; nFlush > 0 && r >= 2 && len(p) == r*BlockSize {
			h.processLeafBatch(p, r)
			p = p[:0]
		}

		// Buffer the tail.
		h.bufferTail(p)
		return n, nil
	}

	// Streaming path: accumulate in buf, flush in whole flush units.
	h.bufferTail(p)
	if processable := len(h.buf) / BlockSize; processable >= lanes {
		nFlush := (processable / lanes) * lanes
		h.processLeafBatch(h.buf[:nFlush*BlockSize], nFlush)
		remaining := copy(h.buf, h.buf[nFlush*BlockSize:])
		h.buf = h.buf[:remaining]
	}
	return n, nil
}

// bufferTail appends p to the leaf buffer. The first fill keeps append's
// exact sizing so one-shot tails stay small; any later growth jumps straight
// to the streaming high-water mark — one whole flush unit plus the partial
// chunk in progress — so steady-state streaming settles after a single
// growth instead of re-copying the buffer through append's doubling steps.
func (h *Hasher) bufferTail(p []byte) {
	if need := len(h.buf) + len(p); cap(h.buf) != 0 && cap(h.buf) < need && need >= growJumpMin {
		h.growBuf(need)
	}
	h.buf = append(h.buf, p...)
}

// growBuf reallocates the leaf buffer with capacity for the streaming
// high-water mark (or need, if larger). Kept out of bufferTail so the no-grow
// fast path stays within the inlining budget.
func (h *Hasher) growBuf(need int) {
	grown := make([]byte, len(h.buf), max(need, (streamChunks+1)*BlockSize))
	copy(grown, h.buf)
	h.buf = grown
}

// processLeafBatch computes leaf CVs for nLeaves complete chunks, draining
// them through a pipeline of arch kernels from widest to narrowest. Each
// stage is gated by the scheduling policy in the kt128_leaves_* arch files
// (constants and fuse* functions), which record the measured tradeoffs:
//
//	batch5  5-chunk hybrid batches                  arm64
//	x8      whole 8-leaf batches                    amd64
//	x3      3-chunk hybrid remainder                arm64
//	pair    2-wide remainders up to pairRemainderMax arm64 (remaining), amd64 (=2)
//	run     3..4-leaf remainders in one YMM quad pass, amd64
//	        5..7 in one masked 8-wide pass
//	x1      whatever remains, serially              all
func (h *Hasher) processLeafBatch(data []byte, nLeaves int) {
	idx := 0

	var cvs [256]byte

	// Hybrid pass: drain leaves five at a time where a hybrid scalar/NEON
	// kernel exists (arm64), covering four leaves at 2-wide NEON throughput
	// with a fifth hidden on the scalar pipes. A remainder of one gives back
	// one batch so pairs drain six leaves without a stranded serial pass; a
	// remainder of three stays available for the hybrid x3 kernel below.
	if hasLeafBatch5 {
		n5 := nLeaves / 5
		if n5 > 0 && nLeaves-n5*5 == 1 {
			n5--
		}
		for range n5 {
			off := idx * BlockSize
			if !processLeavesBatch5Arch(data[off:off+5*BlockSize], &cvs) {
				break
			}
			h.final.absorbCVs(cvs[:160])
			idx += 5
		}
	}

	for hasLeafX8 && idx+8 <= nLeaves {
		off := idx * BlockSize
		if !processLeavesArch(data[off:off+8*BlockSize], &cvs) {
			break
		}
		h.final.absorbCVs(cvs[:])
		idx += 8
	}

	// A three-leaf remainder can use the arm64 hybrid x3 kernel: one scalar
	// leaf advances alongside a NEON pair, then finishes after the pair closes.
	if rem := nLeaves - idx; rem == 3 && processLeavesTripleArch(data[idx*BlockSize:nLeaves*BlockSize], &cvs) {
		h.final.absorbCVs(cvs[:96])
		idx += 3
	}

	// Drain a small leaf remainder in native 2-wide pairs where a pair kernel
	// exists, reading directly from the input with no scratch buffer.
	// pairRemainderMax bounds the counts pairs may drain: on arm64 pairs are
	// the fastest narrow option at any remainder, while on amd64 a pair beats
	// the flat masked run pass only for a remainder of exactly two.
	for idx+2 <= nLeaves && nLeaves-idx <= pairRemainderMax {
		off := idx * BlockSize
		if !processLeavesPairArch(data[off:off+2*BlockSize], &cvs) {
			break
		}
		h.final.absorbCVs(cvs[:64])
		idx += 2
	}

	// Drain a 2..7 leaf remainder in one direct-read masked-gather pass where a
	// run kernel exists (amd64 AVX-512), again with no scratch buffer.
	if rem := nLeaves - idx; rem >= 2 && processLeavesRunArch(data[idx*BlockSize:nLeaves*BlockSize], rem, &cvs) {
		h.final.absorbCVs(cvs[:rem*32])
		idx += rem
	}

	// Small remainder via x1: a single leftover leaf after the pair pass, or a
	// 1..7 leaf remainder on platforms without a pair or run kernel.
	for idx < nLeaves {
		var s1 sponge
		off := idx * BlockSize
		leafStateX1(data[off:off+BlockSize], &s1)
		h.final.absorbCV(&s1)
		idx++
	}

	h.leafCount += uint64(nLeaves)
}

// Read squeezes output from the XOF. On the first call, it finalizes absorption.
func (h *Hasher) Read(p []byte) (int, error) {
	if h.state != stateFinalized {
		h.finalize()
		h.state = stateFinalized
	}
	h.final.squeeze(p)
	return len(p), nil
}

// finalize absorbs the customization suffix and message tail, then applies the
// final pad-and-permute. The suffix C || length_encode(|C|) is built in a small
// scratch buffer so that finalization never reallocates the (possibly large)
// message buffer just to append a few trailing bytes.
func (h *Hasher) finalize() {
	var scratch [64]byte
	var suffix []byte
	if n := len(h.c) + 9; n <= len(scratch) {
		suffix = customSuffix(scratch[:0], h.c)
	} else {
		suffix = customSuffix(make([]byte, 0, n), h.c)
	}

	h.absorbMessage(suffix)
	h.final.padPermute(h.ds)
}

// Clone returns an independent copy of the Hasher. The original and clone evolve independently.
func (h *Hasher) Clone() *Hasher {
	return &Hasher{
		buf:        slices.Clone(h.buf),
		c:          h.c,
		final:      h.final,
		pending:    h.pending,
		pos:        h.pos,
		leafCount:  h.leafCount,
		pendingLen: h.pendingLen,
		ds:         h.ds,
		state:      h.state,
	}
}

// Reset resets the Hasher to its initial state, preserving the customization
// string passed to New. Like the standard library hash implementations, it
// does not scrub buffered message data from memory.
func (h *Hasher) Reset() {
	h.buf = h.buf[:0]
	h.final.reset()
	h.pos = 0
	h.ds = 0
	h.leafCount = 0
	h.pendingLen = 0 // pending's contents are fully overwritten before reuse
	h.state = stateSingle
}

// Equal returns 1 if h and other represent identical states, 0 otherwise.
func (h *Hasher) Equal(other *Hasher) int {
	var a, b [32]byte
	_, _ = h.Clone().Read(a[:])
	_, _ = other.Clone().Read(b[:])
	return subtle.ConstantTimeCompare(a[:], b[:])
}

// customSuffix appends C || length_encode(|C|) to dst and returns the result.
func customSuffix(dst []byte, c []byte) []byte {
	dst = append(dst, c...)
	return lengthEncode(dst, uint64(len(c)))
}

// startTreeMode switches to tree mode: the final node has absorbed exactly
// BlockSize bytes of S_0, so absorb the KT12 marker after it.
func (h *Hasher) startTreeMode() {
	h.final.absorb(kt12Marker[:])
	h.ds = treeDS
	h.state = stateTree
}

// startTreeModeFused enters tree mode by computing the final node's
// S_0 || marker state and the first n-1 leaves' chain values together in one
// fused kernel pass, where a kernel exists. It requires an untouched Hasher
// and n full chunks contiguous in p, and consumes p[:n*BlockSize]. With
// tailBlocks > 0 the pass also absorbs the trailing partial chunk's first
// tailBlocks whole rate-blocks — p must extend that far — into a pending
// leaf state, consuming those bytes too (recorded in h.pendingLen).
func (h *Hasher) startTreeModeFused(p []byte, n, tailBlocks int) bool {
	var cvs [256]byte
	if tailBlocks > 0 {
		if !processS0LeavesTailArch(p, n, tailBlocks, &h.final, pendingSponge(&h.pending), &cvs) {
			return false
		}
		h.pendingLen = tailBlocks * rate
	} else if !processS0LeavesArch(p[:n*BlockSize], n, &h.final, &cvs) {
		return false
	}
	h.ds = treeDS
	h.state = stateTree
	h.final.absorbCVs(cvs[32 : n*32])
	h.leafCount += uint64(n - 1)
	return true
}

// extendPending continues the partially-absorbed trailing leaf left by a
// fused first write. A later write means it is no longer necessarily the
// trailing leaf: if p completes it, it is finished serially — the buffered
// ragged remnant first, then bytes from p — and its chain value absorbed,
// restoring the invariant that the leaf buffer starts at a leaf boundary.
// Returns the unconsumed rest of p; if the leaf remains incomplete, p is
// buffered as more of its remnant and the result is empty.
func (h *Hasher) extendPending(p []byte) []byte {
	pending := pendingSponge(&h.pending)
	room := BlockSize - h.pendingLen - len(h.buf)
	if len(p) < room {
		h.bufferTail(p)
		return nil
	}
	pending.absorb(h.buf)
	pending.absorb(p[:room])
	pending.padPermute(leafDS)
	h.final.absorbCV(pending)
	h.leafCount++
	h.pendingLen = 0
	h.buf = h.buf[:0]
	return p[room:]
}

// absorbMessage absorbs the rest of the logical message into h.final, setting
// h.ds. Message bytes up to one chunk are already in h.final, so in single-node
// mode only the suffix remains, and it decides whether the input fits a single
// node. In tree mode, the buffered leaf tail and the suffix are processed as a
// single byte stream without concatenating or copying them. It does not modify
// h.buf.
func (h *Hasher) absorbMessage(suffix []byte) {
	if h.state == stateSingle {
		room := BlockSize - int(h.pos)
		if len(suffix) <= room {
			// Single-node: KT128 single-node finalization.
			h.ds = singleDS
			h.final.absorb(suffix)
			return
		}

		// The suffix pushes the input past one chunk: complete S_0 from it
		// and enter tree mode; the remainder becomes leaf data.
		h.final.absorb(suffix[:room])
		suffix = suffix[room:]
		h.startTreeMode()
	}

	buf := h.buf

	// A pending trailing leaf from a fused first write: the remaining
	// logical data is its ragged remnant (all of buf) followed by the
	// suffix, absorbed straight into the exported leaf state.
	if h.pendingLen > 0 {
		pending := pendingSponge(&h.pending)
		pending.absorb(buf)
		// The pending leaf takes as much of the suffix as fits; any
		// remainder forms the last leaves.
		n := min(BlockSize-h.pendingLen-len(buf), len(suffix))
		pending.absorb(suffix[:n])
		pending.padPermute(leafDS)
		h.final.absorbCV(pending)
		h.leafCount++
		h.absorbContiguousLeaves(suffix[n:])
	} else {
		// Tree mode: process buf || suffix as leaves S_1, S_2, ... plus terminator.
		// Complete leaves lying entirely within buf use the SIMD batch path
		// directly; head holds the trailing < BlockSize message bytes, so the
		// remaining logical data after them is head || suffix.
		nFull := len(buf) / BlockSize
		head := buf[nFull*BlockSize:]

		// Partial-leaf fusion: when the remaining data forms a single partial
		// leaf, fold an arch-chosen count of trailing complete leaves and the
		// partial leaf's whole rate-blocks into one kernel pass; leading leaves
		// take the batch path.
		if n := fuseTailChunks(nFull, len(head)/rate); n > 0 && len(head)+len(suffix) < BlockSize {
			if lead := nFull - n; lead > 0 {
				h.processLeafBatch(buf[:lead*BlockSize], lead)
			}
			h.fuseTrailingLeaves(buf[(nFull-n)*BlockSize:], n, head, suffix)
		} else {
			if nFull > 0 {
				h.processLeafBatch(buf[:nFull*BlockSize], nFull)
			}
			h.absorbTailLeaves(head, suffix)
		}
	}

	// Terminator: LengthEncode(leafCount) || 0xFF || 0xFF.
	var leBuf [9]byte
	h.final.absorb(lengthEncode(leBuf[:0], h.leafCount))
	h.final.absorb(treeTerminator[:])
}

// fuseTrailingLeaves processes the final n complete leaves and the trailing
// partial leaf head || suffix together: the complete leaves and the partial
// leaf's whole rate-blocks share one kernel pass, and the partial leaf's
// ragged tail and padding finish in Go from the kernel-exported state.
// trailing holds the n complete chunks followed by head; head and suffix
// together must be less than BlockSize bytes.
func (h *Hasher) fuseTrailingLeaves(trailing []byte, n int, head, suffix []byte) {
	nShared := len(head) / rate
	var cvs [256]byte
	var s sponge
	processLeavesTailArch(trailing, n, nShared, &cvs, &s)
	h.final.absorbCVs(cvs[:n*32])
	s.absorb(head[nShared*rate:])
	s.absorb(suffix)
	s.padPermute(leafDS)
	h.final.absorbCV(&s)
	h.leafCount += uint64(n) + 1
}

// absorbTailLeaves processes the final leaves of the logical stream head || tail,
// where head is the trailing < BlockSize message bytes and tail is the remaining
// customization suffix. The single leaf that straddles the head/tail boundary is
// absorbed incrementally so neither slice is copied.
func (h *Hasher) absorbTailLeaves(head, tail []byte) {
	if len(head) == 0 {
		// Remaining data is contiguous in tail.
		h.absorbContiguousLeaves(tail)
		return
	}

	// The straddling leaf takes as much of tail as fits: all of it when
	// head || tail forms a single final partial leaf, leaving nothing for the
	// contiguous pass below.
	n := min(BlockSize-len(head), len(tail))
	var s sponge
	s.absorb(head)
	s.absorb(tail[:n])
	s.padPermute(leafDS)
	h.final.absorbCV(&s)
	h.leafCount++
	h.absorbContiguousLeaves(tail[n:])
}

// absorbContiguousLeaves processes data as zero or more full leaves followed by
// an optional final partial leaf, feeding each chain value into h.final.
func (h *Hasher) absorbContiguousLeaves(data []byte) {
	nFull := len(data) / BlockSize
	if nFull > 0 {
		h.processLeafBatch(data[:nFull*BlockSize], nFull)
	}
	if partial := len(data) - nFull*BlockSize; partial > 0 {
		var s sponge
		leafStateX1(data[nFull*BlockSize:], &s)
		h.final.absorbCV(&s)
		h.leafCount++
	}
}

// kt12Marker is the 8-byte KangarooTwelve marker written after S_0.
var kt12Marker = [8]byte{0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}

// treeTerminator is the two-byte suffix absorbed after LengthEncode(leafCount).
var treeTerminator = [2]byte{0xFF, 0xFF}

// leafStateX1 computes a single KT128 leaf state.
func leafStateX1(data []byte, s *sponge) {
	s.reset()
	s.absorbAll(data, leafDS)
}

// lengthEncode encodes x as in KangarooTwelve (RFC 9861 Section 2.3.1):
// big-endian with no leading zeros, followed by a byte giving the length
// of the encoding. The result is appended to buf and returned as a slice.
func lengthEncode(b []byte, value uint64) []byte {
	if value == 0 {
		return append(b, 0x00)
	}

	n := 8 - (bits.LeadingZeros64(value|1) / 8)
	value <<= (8 - n) * 8
	for range n {
		b = append(b, byte(value>>56))
		value <<= 8
	}
	b = append(b, byte(n))
	return b
}

var _ hash.XOF = (*Hasher)(nil)

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
	buf       []byte // buffered leaf data (tree mode only)
	c         []byte // owned copy of the customization string
	final     sponge // final-node sponge state
	pos       uint64 // total bytes written via Write
	leafCount uint64 // total leaf CVs written to final so far
	state     uint8  // lifecycle: stateSingle -> stateTree -> stateFinalized
	ds        byte   // KT128 customization byte for finalization (singleDS or treeDS)
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

	if h.state == stateSingle {
		// Fused fast path: with S_0 and at least one full leaf contiguous in
		// p and nothing absorbed yet, process them together in one fused
		// kernel pass. Each arch decides how many chunks to take (and when
		// fusion would strand leaves in the buffer) via fuseS0Chunks.
		nFuse := 0
		if h.pos == 0 {
			nFuse = fuseS0Chunks(len(p)/BlockSize, len(p)%BlockSize)
		}
		if nFuse >= 2 && h.startTreeModeFused(p, nFuse) {
			// The rest of p is ordinary leaf data.
			p = p[nFuse*BlockSize:]
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
				h.buf = append(h.buf, p[:need]...)
				p = p[need:]
			}
			h.processLeafBatch(h.buf, len(h.buf)/BlockSize)
			h.buf = h.buf[:0]
		}

		// Flush whole flush-unit multiples in place. An odd leftover chunk is
		// buffered rather than processed here: it costs an x1 pass now or at
		// finalization either way, but a later write may pair it. Complete
		// message leaves need no lookahead, since the customization suffix
		// added at finalization is always non-empty.
		processable := len(p) / BlockSize
		if nFlush := processable - processable%flush; nFlush > 0 {
			h.processLeafBatch(p[:nFlush*BlockSize], nFlush)
			p = p[nFlush*BlockSize:]
		}

		// Buffer the tail.
		h.buf = append(h.buf, p...)
		return n, nil
	}

	// Streaming path: accumulate in buf, flush in whole flush units.
	h.buf = append(h.buf, p...)
	if processable := len(h.buf) / BlockSize; processable >= lanes {
		nFlush := (processable / lanes) * lanes
		h.processLeafBatch(h.buf[:nFlush*BlockSize], nFlush)
		remaining := copy(h.buf, h.buf[nFlush*BlockSize:])
		h.buf = h.buf[:remaining]
	}
	return n, nil
}

// processLeafBatch computes leaf CVs for nLeaves complete chunks, draining
// them through a pipeline of arch kernels from widest to narrowest. Each
// stage is gated by the scheduling policy in the kt128_leaves_* arch files
// (constants and fuse* functions), which record the measured tradeoffs:
//
//	batch5  parity-matched 5-chunk hybrid batches   arm64
//	x8      whole 8-leaf batches                    amd64
//	pair    2-wide remainders up to pairRemainderMax arm64 (any), amd64 (=2)
//	run     2..7-leaf remainders in one masked pass amd64
//	x1      whatever remains, serially              all
func (h *Hasher) processLeafBatch(data []byte, nLeaves int) {
	idx := 0

	var cvs [256]byte

	// Hybrid pass: drain leaves five at a time where a hybrid scalar/NEON
	// kernel exists (arm64), covering four leaves at 2-wide NEON throughput
	// with a fifth hidden on the scalar pipes. The batch count is
	// parity-matched to nLeaves so the remainder stays even (up to 8): the
	// pair loop drains it without a stranded serial leaf, which would cost
	// the same NEON time as the extra hybrid batch saved.
	if hasLeafBatch5 {
		n5 := nLeaves / 5
		if (nLeaves-n5*5)%2 != 0 {
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
		processLeaves(data[off:off+8*BlockSize], &cvs)
		h.final.absorbCVs(cvs[:])
		idx += 8
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
		buf:       slices.Clone(h.buf),
		c:         h.c,
		final:     h.final,
		pos:       h.pos,
		leafCount: h.leafCount,
		ds:        h.ds,
		state:     h.state,
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
// and n full chunks contiguous in p, and consumes p[:n*BlockSize].
func (h *Hasher) startTreeModeFused(p []byte, n int) bool {
	var cvs [256]byte
	if !processS0LeavesArch(p[:n*BlockSize], n, &h.final, &cvs) {
		return false
	}
	h.ds = treeDS
	h.state = stateTree
	h.final.absorbCVs(cvs[32 : n*32])
	h.leafCount += uint64(n - 1)
	return true
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

	if len(head)+len(tail) < BlockSize {
		// A single final partial leaf = head || tail.
		var s sponge
		s.absorb(head)
		s.absorb(tail)
		s.padPermute(leafDS)
		h.final.absorbCV(&s)
		h.leafCount++
		return
	}

	// The straddling leaf is full-size: head || tail[:BlockSize-len(head)].
	n := BlockSize - len(head)
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

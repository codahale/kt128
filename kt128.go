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
		// Fused fast path: with S_0 and the first leaf contiguous in p and
		// nothing absorbed yet, process both together in one x2 kernel pass.
		// Skipped when the leaves after S_0 form whole SIMD-width batches:
		// consuming one would strand lanes-1 of them in the buffer instead of
		// flushing them all directly from p.
		leaves := (len(p) - BlockSize) / BlockSize
		fusable := h.pos == 0 && leaves >= 1 &&
			(leaves < availableLanes || leaves%availableLanes != 0)
		if fusable && h.startTreeModeFused(p) {
			// The rest of p is ordinary leaf data.
			p = p[2*BlockSize:]
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

	lanes := availableLanes

	// Large-write fast path: process chunks directly from p to avoid copying.
	if len(p) >= lanes*BlockSize {
		// Drain any buffered data: flush complete blocks, then complete the
		// partial tail with bytes from p.
		if len(h.buf) > 0 {
			if full := len(h.buf) / BlockSize; full > 0 {
				h.processLeafBatch(h.buf[:full*BlockSize], full)
				remaining := copy(h.buf, h.buf[full*BlockSize:])
				h.buf = h.buf[:remaining]
			}
			if len(h.buf) > 0 {
				need := BlockSize - len(h.buf)
				h.buf = append(h.buf, p[:need]...)
				p = p[need:]
				h.processLeafBatch(h.buf[:BlockSize], 1)
				h.buf = h.buf[:0]
			}
		}

		// The customization suffix added at finalization is always non-empty, so
		// complete message leaves do not need to be retained as lookahead.
		for {
			processable := len(p) / BlockSize
			nFlush := (processable / lanes) * lanes
			if nFlush == 0 {
				break
			}
			h.processLeafBatch(p[:nFlush*BlockSize], nFlush)
			p = p[nFlush*BlockSize:]
		}

		// Buffer the tail.
		h.buf = append(h.buf, p...)
		return n, nil
	}

	// Streaming path: accumulate in buf, flush in SIMD-width batches.
	h.buf = append(h.buf, p...)
	for {
		processable := len(h.buf) / BlockSize
		nFlush := (processable / lanes) * lanes
		if nFlush == 0 {
			break
		}
		h.processLeafBatch(h.buf[:nFlush*BlockSize], nFlush)
		remaining := copy(h.buf, h.buf[nFlush*BlockSize:])
		h.buf = h.buf[:remaining]
	}
	return n, nil
}

// processLeafBatch computes leaf CVs for nLeaves complete chunks using fused SIMD leaf processing.
func (h *Hasher) processLeafBatch(data []byte, nLeaves int) {
	idx := 0

	var cvs [256]byte
	for idx+8 <= nLeaves {
		off := idx * BlockSize
		processLeaves(data[off:off+8*BlockSize], &cvs)
		h.final.absorbCVs(cvs[:])
		idx += 8
	}

	// Drain a 2..7 leaf remainder in native 2-wide pairs where a pair kernel
	// exists (arm64), reading directly from the input with no scratch buffer.
	// Padding to 8 here would copy and zero a 64 KiB buffer only to discard the
	// unused lanes' output.
	for idx+2 <= nLeaves {
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

	// Padded x8 fallback for platforms without a pair or run kernel (amd64 AVX2,
	// other): pad to 8 and use the fused path when utilization is high enough.
	if rem := nLeaves - idx; rem >= 5 {
		off := idx * BlockSize
		var padData [8 * BlockSize]byte
		copy(padData[:rem*BlockSize], data[off:off+rem*BlockSize])
		processLeaves(padData[:], &cvs)
		h.final.absorbCVs(cvs[:rem*32])
		idx += rem
	}

	// Small remainder via x1: a single leftover leaf after the pair pass, or a
	// 1..4 leaf remainder on platforms without a pair kernel.
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
// string passed to New.
func (h *Hasher) Reset() {
	clear(h.buf)
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
// S_0 || marker state and the first leaf's chain value together in one fused
// x2 pass, where a kernel exists. It requires an untouched Hasher and two
// full chunks contiguous in p, and consumes p[:2*BlockSize].
func (h *Hasher) startTreeModeFused(p []byte) bool {
	var cv [32]byte
	if !processS0LeafPairArch(p[:2*BlockSize], &h.final, &cv) {
		return false
	}
	h.ds = treeDS
	h.state = stateTree
	h.final.absorbCVs(cv[:])
	h.leafCount++
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
	// Complete leaves lying entirely within buf use the SIMD batch path directly.
	if nFull := len(buf) / BlockSize; nFull > 0 {
		h.processLeafBatch(buf[:nFull*BlockSize], nFull)
		buf = buf[nFull*BlockSize:]
	}
	// buf now holds the trailing < BlockSize message bytes; the remaining logical
	// data is buf || suffix.
	h.absorbTailLeaves(buf, suffix)

	// Terminator: LengthEncode(leafCount) || 0xFF || 0xFF.
	var leBuf [9]byte
	h.final.absorb(lengthEncode(leBuf[:0], h.leafCount))
	h.final.absorb(treeTerminator[:])
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

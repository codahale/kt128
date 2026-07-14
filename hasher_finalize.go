package kt128

import "math/bits"

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

func customSuffix(dst []byte, c []byte) []byte {
	dst = append(dst, c...)
	return lengthEncode(dst, uint64(len(c)))
}

// startTreeMode switches to tree mode: the final node has absorbed exactly
// BlockSize bytes of S_0, so absorb the KT12 marker after it.

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

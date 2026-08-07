package kt128

import "math/bits"

func (h *state) read(p []byte) (int, error) {
	if h.phase != stateFinalized {
		h.finalize()
		h.phase = stateFinalized
	}
	h.final.squeeze(p)
	return len(p), nil
}

// finalize absorbs the customization string and its length encoding as separate
// segments, then applies the final pad-and-permute.
func (h *state) finalize() {
	h.finalizeWithCustomization(h.c)
}

func (h *state) finalizeWithCustomization(custom []byte) {
	var encoded [9]byte
	h.absorbCustomization(custom, lengthEncode(encoded[:0], uint64(len(custom))))
	var ds uint8 = singleDS
	if h.phase != stateSingle {
		ds = treeDS
	}
	h.final.padPermute(ds)
}

// absorbCustomization absorbs the customization string and its length encoding
// into the logical KT128 input. Message bytes up to one chunk are already in
// h.final; tree-mode data continues through the incremental leaf state.
func (h *state) absorbCustomization(custom, encoded []byte) {
	suffixLen := uint64(len(custom)) + uint64(len(encoded))
	if h.phase == stateSingle {
		room := ChunkSize - int(h.pos)
		if len(custom) <= room && len(encoded) <= room-len(custom) {
			// Single-node: KT128 single-node finalization.
			h.final.absorb(custom)
			h.final.absorb(encoded)
			return
		}

		// The customization string and its encoding push the input past one
		// chunk: complete S_0 from custom || encoded and enter tree mode; the
		// remainder becomes leaf data.
		n := min(room, len(custom))
		h.final.absorb(custom[:n])
		custom = custom[n:]
		room -= n
		if room > 0 {
			n = min(room, len(encoded))
			h.final.absorb(encoded[:n])
			encoded = encoded[n:]
		}
		h.startTreeMode()
	}

	// Tree mode: customization data continues the current leaf stream. The
	// non-empty length encoding ensures the logical stream has a final leaf,
	// unless it completes one exactly.
	h.absorbTreeData(custom)
	h.absorbTreeData(encoded)
	if h.leafLen > 0 {
		h.finishLeaf()
	}

	// Terminator: LengthEncode(number of leaves) || 0xFF || 0xFF.
	var leBuf [9]byte
	h.final.absorb(lengthEncode(leBuf[:0], treeLeafCount(h.pos, suffixLen)))
	h.final.absorb(treeTerminator[:])
}

// treeLeafCount returns the number of chunks after S_0 in a tree whose logical
// input is messageLen+suffixLen bytes. The caller ensures that total exceeds
// ChunkSize. Dividing the two segments separately avoids uint64 overflow when
// the suffix carries the combined length across the 2^64-byte boundary.
func treeLeafCount(messageLen, suffixLen uint64) uint64 {
	leaves := messageLen/ChunkSize + suffixLen/ChunkSize
	remainder := messageLen%ChunkSize + suffixLen%ChunkSize
	leaves += remainder / ChunkSize
	if remainder%ChunkSize == 0 {
		leaves--
	}
	return leaves
}

// lengthEncode appends value encoded as specified by KangarooTwelve
// (RFC 9861 Section 2.3.1) to b and returns the resulting slice. The encoding
// is big-endian with no leading zeros, followed by a byte giving its length.
func lengthEncode(b []byte, value uint64) []byte {
	if value == 0 {
		return append(b, 0x00)
	}

	n := 8 - (bits.LeadingZeros64(value) / 8)
	value <<= (8 - n) * 8
	for range n {
		b = append(b, byte(value>>56))
		value <<= 8
	}
	b = append(b, byte(n))
	return b
}

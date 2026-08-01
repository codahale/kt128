package kt128

import "math/bits"

// Read fills p with output from the XOF and returns len(p), nil. The first call,
// including a zero-length call, finalizes the message; subsequent calls continue
// squeezing the same output stream. Read never returns io.EOF.
func (h *Hasher) Read(p []byte) (int, error) {
	if h.state != stateFinalized {
		h.finalize()
		h.state = stateFinalized
	}
	h.final.squeeze(p)
	return len(p), nil
}

// finalize absorbs the customization string and its length encoding as separate
// segments, then applies the final pad-and-permute.
func (h *Hasher) finalize() {
	var encoded [9]byte
	h.absorbCustomization(h.c, lengthEncode(encoded[:0], uint64(len(h.c))))
	var ds uint8 = singleDS
	if h.state != stateSingle {
		ds = treeDS
	}
	h.final.padPermute(ds)
}

// absorbCustomization absorbs the customization string and its length encoding
// into the logical KT128 input. Message bytes up to one chunk are already in
// h.final; tree-mode data continues through the incremental leaf state.
func (h *Hasher) absorbCustomization(custom, encoded []byte) {
	if h.state == stateSingle {
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

	// Terminator: LengthEncode(leafCount) || 0xFF || 0xFF.
	var leBuf [9]byte
	h.final.absorb(lengthEncode(leBuf[:0], h.leafCount))
	h.final.absorb(treeTerminator[:])
}

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

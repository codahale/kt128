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
	h.absorbMessage(h.c, lengthEncode(encoded[:0], uint64(len(h.c))))
	h.final.padPermute(h.ds)
}

// startTreeMode switches to tree mode: the final node has absorbed exactly
// ChunkSize bytes of S_0, so absorb the KT12 marker after it.

func (h *Hasher) absorbMessage(custom, encoded []byte) {
	if h.state == stateSingle {
		room := ChunkSize - int(h.pos)
		if len(custom) <= room && len(encoded) <= room-len(custom) {
			// Single-node: KT128 single-node finalization.
			h.ds = singleDS
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

	buf := h.buf

	// A pending trailing leaf from a fused first write: the remaining
	// logical data is its ragged remnant (all of buf) followed by custom ||
	// encoded, absorbed straight into the exported leaf state.
	if h.pendingLen > 0 {
		pending := pendingSponge(&h.pending)
		pending.absorb(buf)
		room := ChunkSize - h.pendingLen - len(buf)
		n := min(room, len(custom))
		pending.absorb(custom[:n])
		custom = custom[n:]
		room -= n
		n = min(room, len(encoded))
		pending.absorb(encoded[:n])
		encoded = encoded[n:]
		pending.padPermute(leafDS)
		h.final.absorbCV(pending)
		h.leafCount++
		h.absorbContiguousLeafParts(custom, encoded)
	} else {
		// Tree mode: process buf || custom || encoded as leaves S_1, S_2, ...
		// plus terminator. Complete leaves lying entirely within buf use the SIMD
		// batch path directly; head holds the trailing < ChunkSize message bytes.
		nFull := len(buf) / ChunkSize
		head := buf[nFull*ChunkSize:]

		// Partial-leaf fusion: when the remaining data forms a single partial
		// leaf, fold an arch-chosen count of trailing complete leaves and the
		// partial leaf's whole rate-blocks into one kernel pass; leading leaves
		// take the batch path.
		remaining := ChunkSize - len(head)
		fitsPartial := len(custom) < remaining && len(encoded) < remaining-len(custom)
		if n := fuseTailChunks(nFull, len(head)/rate); n > 0 && fitsPartial {
			if lead := nFull - n; lead > 0 {
				h.processLeafBatch(buf[:lead*ChunkSize], lead)
			}
			h.fuseTrailingLeaves(buf[(nFull-n)*ChunkSize:], n, head, custom, encoded)
		} else {
			if nFull > 0 {
				h.processLeafBatch(buf[:nFull*ChunkSize], nFull)
			}
			h.absorbTailLeaves(head, custom, encoded)
		}
	}

	// Terminator: LengthEncode(leafCount) || 0xFF || 0xFF.
	var leBuf [9]byte
	h.final.absorb(lengthEncode(leBuf[:0], h.leafCount))
	h.final.absorb(treeTerminator[:])
}

// fuseTrailingLeaves processes the final n complete leaves and the trailing
// partial leaf head || custom || encoded together: the complete leaves and the
// partial leaf's whole rate-blocks share one kernel pass, and the partial leaf's
// ragged tail and padding finish in Go from the kernel-exported state. trailing
// holds the n complete chunks followed by head; head, custom, and encoded
// together must be less than ChunkSize bytes.

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

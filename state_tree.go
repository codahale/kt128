package kt128

// startTreeMode switches to tree mode after the final node has absorbed exactly
// ChunkSize bytes of S_0.
func (h *state) startTreeMode() {
	h.final.absorb(kt12Marker[:])
	h.phase = stateTree
}

// startTreeModeFused enters tree mode by computing the final node's
// S_0 || marker state and the first n-1 leaves' chain values together in one
// fused kernel pass, where a kernel exists. It requires an untouched state
// and n full chunks contiguous in p, and consumes p[:n*ChunkSize]. With
// tailBlocks > 0 the pass also absorbs the trailing partial chunk's first
// tailBlocks whole rate-blocks into h.leaf.
func (h *state) startTreeModeFused(p []byte, n, tailBlocks int) bool {
	var cvs [256]byte
	if tailBlocks > 0 {
		if !tryProcessS0LeavesTailArch(p, n, tailBlocks, &h.final, &h.leaf, &cvs) {
			return false
		}
		h.leafLen = tailBlocks * rate
	} else if !tryProcessS0LeavesArch(p[:n*ChunkSize], n, &h.final, &cvs) {
		return false
	}
	h.phase = stateTree
	h.final.absorbCVs(cvs[32 : n*32])
	return true
}

// startLeafFused attempts to process n complete leaves and the whole rate blocks
// of tail together, leaving the partial leaf ready for subsequent writes. It
// returns false without changing h when the scheduled kernel is unavailable.
func (h *state) startLeafFused(trailing []byte, n int, tail []byte) bool {
	nShared := len(tail) / rate
	var cvs [256]byte
	if !tryProcessLeavesTailArch(trailing, n, nShared, &cvs, &h.leaf) {
		return false
	}
	h.final.absorbCVs(cvs[:n*32])
	h.leaf.absorb(tail[nShared*rate:])
	h.leafLen = len(tail)
	return true
}

// finishLeaf finalizes the current leaf and absorbs its chain value.
func (h *state) finishLeaf() {
	h.leaf.padPermute(leafDS)
	h.final.absorbCV(&h.leaf)
	h.leaf.reset()
	h.leafLen = 0
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

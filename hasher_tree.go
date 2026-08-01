package kt128

// startTreeMode switches to tree mode after the final node has absorbed exactly
// ChunkSize bytes of S_0.
func (h *Hasher) startTreeMode() {
	h.final.absorb(kt12Marker[:])
	h.state = stateTree
}

// startTreeModeFused enters tree mode by computing the final node's
// S_0 || marker state and the first n-1 leaves' chain values together in one
// fused kernel pass, where a kernel exists. It requires an untouched Hasher
// and n full chunks contiguous in p, and consumes p[:n*ChunkSize]. With
// tailBlocks > 0 the pass also absorbs the trailing partial chunk's first
// tailBlocks whole rate-blocks into h.leaf.
func (h *Hasher) startTreeModeFused(p []byte, n, tailBlocks int) bool {
	var cvs [256]byte
	defer wipeBytes(cvs[:])
	if tailBlocks > 0 {
		if !processS0LeavesTailArch(p, n, tailBlocks, &h.final, &h.leaf, &cvs) {
			return false
		}
		h.leafLen = tailBlocks * rate
	} else if !processS0LeavesArch(p[:n*ChunkSize], n, &h.final, &cvs) {
		return false
	}
	h.state = stateTree
	h.final.absorbCVs(cvs[32 : n*32])
	h.leafCount += uint64(n - 1)
	return true
}

// startLeafFused processes n complete leaves and the whole rate blocks of tail
// together, leaving the partial leaf ready for subsequent writes.
func (h *Hasher) startLeafFused(trailing []byte, n int, tail []byte) {
	nShared := len(tail) / rate
	var cvs [256]byte
	defer wipeBytes(cvs[:])
	processLeavesTailArch(trailing, n, nShared, &cvs, &h.leaf)
	h.final.absorbCVs(cvs[:n*32])
	h.leaf.absorb(tail[nShared*rate:])
	h.leafCount += uint64(n)
	h.leafLen = len(tail)
}

// finishLeaf finalizes the current leaf and absorbs its chain value.
func (h *Hasher) finishLeaf() {
	h.leaf.padPermute(leafDS)
	h.final.absorbCV(&h.leaf)
	h.leafCount++
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

// lengthEncode encodes x as in KangarooTwelve (RFC 9861 Section 2.3.1):
// big-endian with no leading zeros, followed by a byte giving the length
// of the encoding. The result is appended to buf and returned as a slice.

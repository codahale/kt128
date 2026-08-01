package kt128

func (h *Hasher) startTreeMode() {
	h.final.absorb(kt12Marker[:])
	h.ds = treeDS
	h.state = stateTree
}

// startTreeModeFused enters tree mode by computing the final node's
// S_0 || marker state and the first n-1 leaves' chain values together in one
// fused kernel pass, where a kernel exists. It requires an untouched Hasher
// and n full chunks contiguous in p, and consumes p[:n*ChunkSize]. With
// tailBlocks > 0 the pass also absorbs the trailing partial chunk's first
// tailBlocks whole rate-blocks — p must extend that far — into a pending
// leaf state, consuming those bytes too (recorded in h.pendingLen).
func (h *Hasher) startTreeModeFused(p []byte, n, tailBlocks int) bool {
	var cvs [256]byte
	defer wipeBytes(cvs[:])
	if tailBlocks > 0 {
		if !processS0LeavesTailArch(p, n, tailBlocks, &h.final, pendingSponge(&h.pending), &cvs) {
			return false
		}
		h.pendingLen = tailBlocks * rate
	} else if !processS0LeavesArch(p[:n*ChunkSize], n, &h.final, &cvs) {
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

func (h *Hasher) fuseTrailingLeaves(trailing []byte, n int, head, custom, encoded []byte) {
	nShared := len(head) / rate
	var cvs [256]byte
	var s sponge
	defer wipeBytes(cvs[:])
	defer s.wipe()
	processLeavesTailArch(trailing, n, nShared, &cvs, &s)
	h.final.absorbCVs(cvs[:n*32])
	s.absorb(head[nShared*rate:])
	s.absorb(custom)
	s.absorb(encoded)
	s.padPermute(leafDS)
	h.final.absorbCV(&s)
	h.leafCount += uint64(n) + 1
}

// absorbTailLeaves processes the final leaves of the logical stream
// head || custom || encoded. The single leaf that straddles a segment boundary
// is absorbed incrementally so no segment is copied.
func (h *Hasher) absorbTailLeaves(head, custom, encoded []byte) {
	if len(head) == 0 {
		h.absorbContiguousLeafParts(custom, encoded)
		return
	}

	var s sponge
	defer s.wipe()
	s.absorb(head)
	room := ChunkSize - len(head)
	n := min(room, len(custom))
	s.absorb(custom[:n])
	custom = custom[n:]
	room -= n
	n = min(room, len(encoded))
	s.absorb(encoded[:n])
	encoded = encoded[n:]
	s.padPermute(leafDS)
	h.final.absorbCV(&s)
	h.leafCount++
	h.absorbContiguousLeafParts(custom, encoded)
}

// absorbContiguousLeafParts processes custom || encoded as zero or more full
// leaves followed by an optional final partial leaf. Full leaves wholly within
// custom use the SIMD batch path; only the leaf crossing the segment boundary is
// absorbed incrementally.
func (h *Hasher) absorbContiguousLeafParts(custom, encoded []byte) {
	nFull := len(custom) / ChunkSize
	if nFull > 0 {
		h.processLeafBatch(custom[:nFull*ChunkSize], nFull)
		custom = custom[nFull*ChunkSize:]
	}
	if len(custom) == 0 {
		h.absorbContiguousLeaves(encoded)
		return
	}

	var s sponge
	defer s.wipe()
	s.absorb(custom)
	n := min(ChunkSize-len(custom), len(encoded))
	s.absorb(encoded[:n])
	s.padPermute(leafDS)
	h.final.absorbCV(&s)
	h.leafCount++
	h.absorbContiguousLeaves(encoded[n:])
}

// absorbContiguousLeaves processes data as zero or more full leaves followed by
// an optional final partial leaf, feeding each chain value into h.final.
func (h *Hasher) absorbContiguousLeaves(data []byte) {
	nFull := len(data) / ChunkSize
	if nFull > 0 {
		h.processLeafBatch(data[:nFull*ChunkSize], nFull)
	}
	if partial := len(data) - nFull*ChunkSize; partial > 0 {
		var s sponge
		leafStateX1(data[nFull*ChunkSize:], &s)
		h.final.absorbCV(&s)
		s.wipe()
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

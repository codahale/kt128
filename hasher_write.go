package kt128

// Write absorbs p as message data. It always returns len(p), nil and does not
// retain p. Write panics after any call to [Hasher.Read], including a zero-length
// read, if h has been cleared, or if accepting p would bring the message length
// to 2^64 bytes.
func (h *Hasher) Write(p []byte) (int, error) {
	h.checkNotCleared()
	if h.state == stateFinalized {
		panic("kt128: Hasher is finalized")
	}

	n := len(p)
	if uint64(n) > ^uint64(0)-h.pos {
		panic("kt128: message length exceeds 2^64-1 bytes")
	}
	if h.state == stateSingle {
		// Fused fast path: with S_0 and at least one full leaf contiguous in
		// p and nothing absorbed yet, process them together in one fused
		// kernel pass. Each arch decides how many chunks to take via
		// fuseS0Chunks. When the pass consumes every complete chunk, the
		// trailing partial chunk's whole rate-blocks may ride an idle lane
		// (fuseS0TailBlocks), leaving a partially absorbed leaf.
		nFuse, nTail := 0, 0
		if h.pos == 0 {
			chunks, tail := len(p)/ChunkSize, len(p)%ChunkSize
			nFuse = fuseS0Chunks(chunks, tail)
			if nFuse == chunks {
				nTail = fuseS0TailBlocks(chunks, tail)
			}
		}
		if nFuse >= 2 && h.startTreeModeFused(p, nFuse, nTail) {
			// The rest of p continues the partial leaf exported by the fused
			// pass, if any, then becomes ordinary leaf data.
			p = p[nFuse*ChunkSize+h.leafLen:]
		} else {
			// Single-node finalization and tree-mode S_0 absorb the first
			// chunk into the final node identically, so message bytes are
			// absorbed eagerly with no buffering; the two modes diverge only
			// once the input exceeds one chunk.
			room := ChunkSize - int(h.pos)
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
	h.absorbTreeData(p)
	return n, nil
}

// absorbTreeData absorbs tree-mode input without retaining its bytes. A partial
// leaf is continued through h.leaf; complete leaves contiguous in p still use
// the SIMD batch path.
func (h *Hasher) absorbTreeData(p []byte) {
	if h.leafLen > 0 {
		n := min(ChunkSize-h.leafLen, len(p))
		h.leaf.absorb(p[:n])
		h.leafLen += n
		p = p[n:]
		if h.leafLen < ChunkSize {
			return
		}
		h.finishLeaf()
	}

	nFull := len(p) / ChunkSize
	tail := p[nFull*ChunkSize:]
	if nFull > 0 && len(tail) > 0 {
		if n := fuseTailChunks(nFull, len(tail)/rate); n > 0 {
			lead := nFull - n
			if lead > 0 {
				h.processLeafBatch(p[:lead*ChunkSize], lead)
			}
			if h.startLeafFused(p[lead*ChunkSize:], n, tail) {
				return
			}
			// A capability change between scheduling and dispatch must not
			// change the digest. The leading leaves are already absorbed; let
			// the ordinary paths process the remaining leaves and tail.
			p = p[lead*ChunkSize:]
			nFull = n
		}
	}

	if nFull > 0 {
		h.processLeafBatch(p[:nFull*ChunkSize], nFull)
	}
	if len(tail) > 0 {
		h.leaf.absorb(tail)
		h.leafLen = len(tail)
	}
}

// processLeafBatch computes leaf CVs for nLeaves complete chunks, draining
// them through a pipeline of arch kernels from widest to narrowest. Each
// stage is gated by the scheduling policy in the leaves_dispatch_* architecture files
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
	defer wipeBytes(cvs[:])

	// Hybrid pass: drain leaves five at a time where a hybrid scalar/NEON
	// kernel exists (arm64), covering four leaves at 2-wide NEON throughput
	// with a fifth hidden on the scalar pipes. A remainder of one gives back
	// one batch so pairs drain six leaves without a stranded serial pass; a
	// remainder of three stays available for the hybrid x3 kernel below.
	n5 := nLeaves / 5
	if n5 > 0 && nLeaves-n5*5 == 1 {
		n5--
	}
	for range n5 {
		off := idx * ChunkSize
		if !tryProcessLeavesBatch5Arch(data[off:off+5*ChunkSize], &cvs) {
			break
		}
		h.final.absorbCVs(cvs[:160])
		idx += 5
	}

	for idx+8 <= nLeaves {
		off := idx * ChunkSize
		if !tryProcessLeavesX8Arch(data[off:off+8*ChunkSize], &cvs) {
			break
		}
		h.final.absorbCVs(cvs[:])
		idx += 8
	}

	// A three-leaf remainder can use the arm64 hybrid x3 kernel: one scalar
	// leaf advances alongside a NEON pair, then finishes after the pair closes.
	if rem := nLeaves - idx; rem == 3 && tryProcessLeavesTripleArch(data[idx*ChunkSize:nLeaves*ChunkSize], &cvs) {
		h.final.absorbCVs(cvs[:96])
		idx += 3
	}

	// Drain a small leaf remainder in native 2-wide pairs where a pair kernel
	// exists, reading directly from the input with no scratch buffer.
	// pairRemainderMax bounds the counts pairs may drain: on arm64 pairs are
	// the fastest narrow option at any remainder, while on amd64 a pair beats
	// the flat masked run pass only for a remainder of exactly two.
	for idx+2 <= nLeaves && nLeaves-idx <= pairRemainderMax {
		off := idx * ChunkSize
		if !tryProcessLeavesPairArch(data[off:off+2*ChunkSize], &cvs) {
			break
		}
		h.final.absorbCVs(cvs[:64])
		idx += 2
	}

	// Drain a 2..7 leaf remainder in one direct-read masked-gather pass where a
	// run kernel exists (amd64 AVX-512), again with no scratch buffer.
	if rem := nLeaves - idx; rem >= 2 && tryProcessLeavesRunArch(data[idx*ChunkSize:nLeaves*ChunkSize], rem, &cvs) {
		h.final.absorbCVs(cvs[:rem*32])
		idx += rem
	}

	// Small remainder via x1: a single leftover leaf after the pair pass, or a
	// 1..7 leaf remainder on platforms without a pair or run kernel.
	for idx < nLeaves {
		var s1 sponge
		off := idx * ChunkSize
		leafStateX1(data[off:off+ChunkSize], &s1)
		h.final.absorbCV(&s1)
		s1.wipe()
		idx++
	}
}

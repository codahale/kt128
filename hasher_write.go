package kt128

func (h *Hasher) Write(p []byte) (int, error) {
	if h.state == stateFinalized {
		panic("kt128: Hasher is finalized")
	}

	n := len(p)
	fusedS0 := false

	if h.state == stateSingle {
		// Fused fast path: with S_0 and at least one full leaf contiguous in
		// p and nothing absorbed yet, process them together in one fused
		// kernel pass. Each arch decides how many chunks to take (and when
		// fusion would strand leaves in the buffer) via fuseS0Chunks. When
		// the pass consumes every complete chunk, the trailing partial
		// chunk's whole rate-blocks may ride an idle lane (fuseS0TailBlocks),
		// leaving a pending partially-absorbed leaf.
		nFuse, nTail := 0, 0
		if h.pos == 0 {
			chunks, tail := len(p)/BlockSize, len(p)%BlockSize
			nFuse = fuseS0Chunks(chunks, tail)
			if nFuse == chunks {
				nTail = fuseS0TailBlocks(chunks, tail)
			}
		}
		if nFuse >= 2 && h.startTreeModeFused(p, nFuse, nTail) {
			// The rest of p is ordinary leaf data — or, with a pending
			// leaf, its ragged remnant, buffered by extendPending below.
			p = p[nFuse*BlockSize+h.pendingLen:]
			fusedS0 = true
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

	// A pending trailing leaf holds the stream position: p continues it.
	if h.pendingLen > 0 {
		if p = h.extendPending(p); len(p) == 0 {
			return n, nil
		}
	}

	// A fused first write's chunk-aligned tail drains in place as well: the
	// fused pass makes this bulk traffic ending on a chunk boundary — the
	// same shape the direct path drains in place — but the leftover may sit
	// below the flush-unit gate, and streaming it costs up to seven chunks of
	// memcpy plus an allocation for the same narrow pass at finalization
	// (+11..38% at 80..120 KiB one-shots, Emerald Rapids). A ragged tail
	// still buffers whole so its trailing chunks can ride a fused pass with
	// the partial at finalization.
	if r := len(p) / BlockSize; fusedS0 && r >= 2 && len(p) == r*BlockSize {
		h.processLeafBatch(p, r)
		return n, nil
	}

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
				h.bufferTail(p[:need])
				p = p[need:]
			}

			// Complete a SIMD-width batch from buffered whole chunks before
			// draining it. The bounded copy avoids sending a small buffered
			// remainder through a narrow kernel when this write can fill all
			// lanes.
			if buffered := len(h.buf) / BlockSize; buffered < lanes && len(p) >= (lanes-buffered)*BlockSize {
				take := (lanes - buffered) * BlockSize
				h.bufferTail(p[:take])
				p = p[take:]
			}
			h.processLeafBatch(h.buf, len(h.buf)/BlockSize)
			h.buf = h.buf[:0]
		}

		// Flush the architecture-selected complete-chunk prefix in place.
		// Complete message leaves need no lookahead, since the customization
		// suffix added at finalization is always non-empty.
		processable := len(p) / BlockSize
		nFlush := directFlushChunks(processable)
		if nFlush > 0 {
			h.processLeafBatch(p[:nFlush*BlockSize], nFlush)
			p = p[nFlush*BlockSize:]
		}

		// A chunk-aligned remainder of two or more after a whole-unit flush
		// drains in place as well: with no partial tail to extend, finalization
		// would push these same chunks through the same narrow kernels from the
		// buffer, so buffering buys no better pass and costs the copy plus its
		// allocation (up to seven chunks). A sub-chunk continuation could have
		// completed the buffered chunks into a whole batch, but a stream
		// overwhelmingly ends at a bulk write, and the miss costs one narrow
		// pass. A single trailing chunk still buffers — an x1 pass now or at
		// finalization costs the same, and a later write may pair it — and a
		// ragged tail still buffers whole, so its trailing chunks can ride a
		// fused pass with the partial at finalization.
		if r := len(p) / BlockSize; nFlush > 0 && r >= 2 && len(p) == r*BlockSize {
			h.processLeafBatch(p, r)
			p = p[:0]
		}

		// Buffer the tail.
		h.bufferTail(p)
		return n, nil
	}

	// Streaming path: accumulate in buf, flush in whole flush units.
	h.bufferTail(p)
	if processable := len(h.buf) / BlockSize; processable >= lanes {
		nFlush := (processable / lanes) * lanes
		h.processLeafBatch(h.buf[:nFlush*BlockSize], nFlush)
		remaining := copy(h.buf, h.buf[nFlush*BlockSize:])
		h.buf = h.buf[:remaining]
	}
	return n, nil
}

// bufferTail appends p to the leaf buffer. The first fill keeps append's
// exact sizing so one-shot tails stay small; any later growth jumps straight
// to the streaming high-water mark — one whole flush unit plus the partial
// chunk in progress — so steady-state streaming settles after a single
// growth instead of re-copying the buffer through append's doubling steps.
func (h *Hasher) bufferTail(p []byte) {
	if need := len(h.buf) + len(p); cap(h.buf) != 0 && cap(h.buf) < need && need >= growJumpMin {
		h.growBuf(need)
	}
	h.buf = append(h.buf, p...)
}

// growBuf reallocates the leaf buffer with capacity for the streaming
// high-water mark (or need, if larger). Kept out of bufferTail so the no-grow
// fast path stays within the inlining budget.
func (h *Hasher) growBuf(need int) {
	grown := make([]byte, len(h.buf), max(need, (streamChunks+1)*BlockSize))
	copy(grown, h.buf)
	h.buf = grown
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

	// Hybrid pass: drain leaves five at a time where a hybrid scalar/NEON
	// kernel exists (arm64), covering four leaves at 2-wide NEON throughput
	// with a fifth hidden on the scalar pipes. A remainder of one gives back
	// one batch so pairs drain six leaves without a stranded serial pass; a
	// remainder of three stays available for the hybrid x3 kernel below.
	if hasLeafBatch5 {
		n5 := nLeaves / 5
		if n5 > 0 && nLeaves-n5*5 == 1 {
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
		if !processLeavesArch(data[off:off+8*BlockSize], &cvs) {
			break
		}
		h.final.absorbCVs(cvs[:])
		idx += 8
	}

	// A three-leaf remainder can use the arm64 hybrid x3 kernel: one scalar
	// leaf advances alongside a NEON pair, then finishes after the pair closes.
	if rem := nLeaves - idx; rem == 3 && processLeavesTripleArch(data[idx*BlockSize:nLeaves*BlockSize], &cvs) {
		h.final.absorbCVs(cvs[:96])
		idx += 3
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

func (h *Hasher) extendPending(p []byte) []byte {
	pending := pendingSponge(&h.pending)
	room := BlockSize - h.pendingLen - len(h.buf)
	if len(p) < room {
		h.bufferTail(p)
		return nil
	}
	pending.absorb(h.buf)
	pending.absorb(p[:room])
	pending.padPermute(leafDS)
	h.final.absorbCV(pending)
	h.leafCount++
	h.pendingLen = 0
	h.buf = h.buf[:0]
	return p[room:]
}

// absorbMessage absorbs the rest of the logical message into h.final, setting
// h.ds. Message bytes up to one chunk are already in h.final, so in single-node
// mode only the suffix remains, and it decides whether the input fits a single
// node. In tree mode, the buffered leaf tail and the suffix are processed as a
// single byte stream without concatenating or copying them. It does not modify
// h.buf.

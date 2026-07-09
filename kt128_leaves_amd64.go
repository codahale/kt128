//go:build amd64 && !purego

package kt128

import (
	"unsafe"

	"github.com/codahale/kt128/internal/cpuid"
)

// ─── Scheduling policy ───
//
// amd64 leans on wide masked kernels: whole 8-leaf batches and 2..7-leaf
// remainders run as flat 8-wide AVX-512 passes (gathers cost the same at any
// lane occupancy), with 2-wide XMM pairs taking over wherever only two lanes
// are live. Without AVX-512 the AVX2 x8 and quad kernels cover the batch and
// remainder paths and the narrow-kernel policies report unavailable.

const availableLanes = 8

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss. amd64 has no cheap narrow batch kernel
// (the remainder paths use masked gathers or dummy lanes), so it stays at
// the full SIMD width.
const flushChunks = 8

// streamChunks is the streaming-path flush unit; amd64 has no hybrid batch
// kernel, so it is the SIMD width.
const streamChunks = availableLanes

// hasLeafX8 reports that amd64 drains whole 8-leaf batches through a
// dedicated 8-wide kernel.
const hasLeafX8 = true

// hasLeafBatch5 reports that amd64 has no hybrid scalar/SIMD batch kernel;
// with 16 general-purpose registers a woven scalar lane would spill heavily.
const hasLeafBatch5 = false

// pairRemainderMax bounds the leaf counts the pair loop may drain: a 2-wide
// XMM pass costs ~0.63 of a flat masked pass (Emerald Rapids), so a pair
// wins for a remainder of exactly two but chains of pairs lose to the
// masked run kernel from three up.
const pairRemainderMax = 2

// fuseS0Chunks returns how many chunks (S_0 plus leaves) the fused kernel
// should consume from a first write containing the given number of full
// chunks (plus tail trailing bytes), or 0 to skip fusion. Up to
// availableLanes chunks fuse into one pass (2-wide XMM at exactly two,
// 8-wide masked above), replacing the serial S_0 absorption: at or below one
// pass this is a strict win, and above it the 1..7 chunks the fused pass
// strands past the last whole batch drain in one fused pass at finalization
// (tail fusion), so fusing still saves the serial S_0 pass outright. The
// exception is a chunk count of 1 mod availableLanes with less than a whole
// rate-block of tail: the stranded leaf then has no partial to pair with,
// its serial pass cancels the saving, and fusion only adds a buffer copy.
func fuseS0Chunks(chunks, tail int) int {
	if !cpuid.HasAVX512 || chunks < 2 {
		return 0
	}
	if chunks <= availableLanes || chunks%availableLanes != 1 || tail >= rate {
		return min(chunks, availableLanes)
	}
	return 0
}

// fuseTailChunks returns how many trailing complete leaves finalization
// should fold into one pass with the partial leaf's whole rate-blocks, or 0
// to keep the serial path. On AVX-512 the tail rides a 2..7-leaf remainder
// batch as an extra masked gather lane essentially free; whole multiples of
// 8 fill all lanes and drain through the x8 kernel instead. A single
// leftover leaf pairs with the tail in a 2-wide XMM pass: the masked run
// kernel loses at 2 lanes (its gather absorb only amortizes over enough
// lanes) but the pair pass costs well under the two serial x1 passes it
// replaces.
func fuseTailChunks(nFull, nShared int) int {
	if !cpuid.HasAVX512 || nShared == 0 {
		return 0
	}
	return nFull % 8
}

// ─── Kernels ───

//go:noescape
func processLeavesAVX512(input *byte, cvs *byte)

//go:noescape
func processLeavesAVX2(input *byte, cvs *byte)

//go:noescape
func processLeavesQuadAVX2(in0, in1, in2, in3, cvs *byte)

//go:noescape
func processLeavesRunAVX512(input *byte, cvs *byte, n uint64)

//go:noescape
func processLeavesRunPartialAVX512(input *byte, cvs *byte, n, nShared uint64, lane1 *uint64)

//go:noescape
func processLeavesPairAVX512(input *byte, cvs *byte)

//go:noescape
func processS0LeafPairAVX512(input *byte, state *uint64, cv *byte)

//go:noescape
func processLeafPairPartialAVX512(in0, in1 *byte, nShared uint64, cv *byte, lane1 *uint64)

//go:noescape
func processS0LeavesAVX512(input *byte, state *uint64, cvs *byte, n uint64)

// ─── Kernel wrappers ───

func processLeavesArch(input []byte, cvs *[256]byte) bool {
	if cpuid.HasAVX512 {
		processLeavesAVX512(unsafe.SliceData(input), &cvs[0])
	} else {
		processLeavesAVX2(unsafe.SliceData(input), &cvs[0])
	}
	return true
}

func processLeavesBatch5Arch(_ []byte, _ *[256]byte) bool { return false }

// processLeavesPairArch computes 2 leaf CVs from 2 contiguous chunks via a
// single 2-wide XMM pass, reading directly from the input with plain loads.
func processLeavesPairArch(input []byte, cvs *[256]byte) bool {
	if !cpuid.HasAVX512 {
		return false
	}
	processLeavesPairAVX512(unsafe.SliceData(input), &cvs[0])
	return true
}

// processLeavesRunArch computes n (2..7) leaf CVs by reading the chunks directly
// with no scratch buffer: a single 8-wide masked-gather pass on AVX-512, or one
// to two x4 passes with dummy lanes on AVX2.
func processLeavesRunArch(data []byte, n int, cvs *[256]byte) bool {
	if cpuid.HasAVX512 {
		processLeavesRunAVX512(unsafe.SliceData(data), &cvs[0], uint64(n))
		return true
	}

	// AVX2: run x4 passes, pointing dummy lanes at an in-bounds chunk. The first
	// pass covers leaves 0..3, the second (only when n > 4) covers 4..n-1. CVs
	// for dummy lanes are written but never read by the caller.
	base := unsafe.Pointer(unsafe.SliceData(data))
	var p [8]*byte
	for i := range p {
		off := 0
		if i < n {
			off = i * BlockSize // dummy lanes (i >= n) read chunk 0, discarded
		}
		p[i] = (*byte)(unsafe.Add(base, off))
	}
	processLeavesQuadAVX2(p[0], p[1], p[2], p[3], &cvs[0])
	if n > 4 {
		processLeavesQuadAVX2(p[4], p[5], p[6], p[7], &cvs[128])
	}
	return true
}

// processS0LeavesArch fuses the final node's absorption of S_0 || kt12 marker
// with leaf compression in one pass: a 2-wide XMM pair at n == 2 (pair cost
// instead of a flat masked pass) or an 8-wide AVX-512 pass for n in 3..8.
// input must be n*BlockSize contiguous bytes (S_0 then n-1 leaves) and final
// must be a zero sponge. On return, final holds the state after
// S_0 || marker and cvs[32:n*32] the leaves' chain values.
func processS0LeavesArch(input []byte, n int, final *sponge, cvs *[256]byte) bool {
	if !cpuid.HasAVX512 || n < 2 || n > 8 {
		return false
	}
	if n == 2 {
		processS0LeafPairAVX512(unsafe.SliceData(input), &final.a[0], &cvs[32])
	} else {
		processS0LeavesAVX512(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n))
	}
	final.pos = BlockSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	return true
}

// processLeavesTailArch computes n (1..7) trailing complete leaf CVs while
// absorbing the following partial leaf's nShared whole rate-blocks into
// partial's state in the same pass — a 2-wide XMM pair for n == 1, a masked
// gather pass otherwise; the caller finishes the partial leaf's ragged tail
// and padding through the sponge. trailing must hold the n complete chunks
// followed contiguously by the partial head.
func processLeavesTailArch(trailing []byte, n, nShared int, cvs *[256]byte, partial *sponge) bool {
	if !cpuid.HasAVX512 {
		return false
	}
	if n == 1 {
		processLeafPairPartialAVX512(unsafe.SliceData(trailing), unsafe.SliceData(trailing[BlockSize:]), uint64(nShared), &cvs[0], &partial.a[0])
		return true
	}
	processLeavesRunPartialAVX512(unsafe.SliceData(trailing), &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
	return true
}

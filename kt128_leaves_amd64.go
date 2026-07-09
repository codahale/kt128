//go:build amd64 && !purego

package kt128

import (
	"unsafe"

	"github.com/codahale/kt128/internal/cpuid"
)

const availableLanes = 8

// streamChunks is the streaming-path flush unit; amd64 has no hybrid batch
// kernel, so it is the SIMD width.
const streamChunks = availableLanes

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss. amd64 has no cheap narrow kernel (the
// remainder paths use masked gathers or dummy lanes), so it stays at the full
// SIMD width.
const flushChunks = 8

//go:noescape
func processLeavesAVX512(input *byte, cvs *byte)

//go:noescape
func processLeavesAVX2(input *byte, cvs *byte)

//go:noescape
func processLeavesRunAVX512(input *byte, cvs *byte, n uint64)

//go:noescape
func processLeavesQuadAVX2(in0, in1, in2, in3, cvs *byte)

func processLeavesArch(input []byte, cvs *[256]byte) bool {
	if cpuid.HasAVX512 {
		processLeavesAVX512(unsafe.SliceData(input), &cvs[0])
	} else {
		processLeavesAVX2(unsafe.SliceData(input), &cvs[0])
	}
	return true
}

// processLeavesPairArch reports that no 2-wide pair kernel is available on
// amd64; the run kernel (AVX-512) or padded x8 path drains remainders instead.
func processLeavesPairArch(_ []byte, _ *[256]byte) bool { return false }

// hasLeafBatch5 reports that amd64 has no hybrid scalar/SIMD batch kernel;
// with 16 general-purpose registers a woven scalar lane would spill heavily.
const hasLeafBatch5 = false

func processLeavesBatch5Arch(_ []byte, _ *[256]byte) bool { return false }

// hasPartialLeafFuse reports that amd64 has no complete+partial pair kernel;
// finalization drains the trailing leaves serially.
const hasPartialLeafFuse = false

func processLeafPairPartialArch(_, _ []byte, _ int, _ *[32]byte, _ *sponge) bool { return false }

//go:noescape
func processS0LeavesAVX512(input *byte, state *uint64, cvs *byte, n uint64)

// processS0LeavesArch fuses the final node's absorption of S_0 || kt12 marker
// with leaf compression in one 8-wide AVX-512 pass. input must be n*BlockSize
// contiguous bytes (S_0 then n-1 leaves, n in 2..8) and final must be a zero
// sponge. On return, final holds the state after S_0 || marker and
// cvs[32:n*32] the leaves' chain values.
func processS0LeavesArch(input []byte, n int, final *sponge, cvs *[256]byte) bool {
	if !cpuid.HasAVX512 || n < 2 || n > 8 {
		return false
	}
	processS0LeavesAVX512(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n))
	final.pos = BlockSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	return true
}

// fuseS0Chunks returns how many chunks (S_0 plus leaves) the fused kernel
// should consume from a first write containing the given number of full
// chunks, or 0 to skip fusion. Up to availableLanes chunks fuse into one
// 8-wide pass. At or below one pass fusion is a strict win: those leaves
// would otherwise be buffered and run-kernel'd at finalization. Above it,
// fusion is taken only when the chunk count is a whole number of passes, so
// consuming availableLanes chunks doesn't leave a larger buffered tail than
// the unfused path.
func fuseS0Chunks(chunks int) int {
	if !cpuid.HasAVX512 || chunks < 2 {
		return 0
	}
	if chunks <= availableLanes || chunks%availableLanes == 0 {
		return min(chunks, availableLanes)
	}
	return 0
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

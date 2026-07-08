//go:build amd64 && !purego

package kt128

import (
	"unsafe"

	"github.com/codahale/kt128/internal/cpuid"
)

const availableLanes = 8

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

// processS0LeafPairArch reports that no fused S_0+leaf kernel is available on
// amd64; S_0 is absorbed through the x1 sponge path instead.
func processS0LeafPairArch(_ []byte, _ *sponge, _ *[32]byte) bool { return false }

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

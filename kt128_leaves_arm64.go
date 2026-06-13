//go:build arm64 && !purego

package kt128

import "unsafe"

const availableLanes = 8

//go:noescape
func processLeavesARM64(input *byte, cvs *byte)

//go:noescape
func processLeavesPairARM64(input *byte, cvs *byte)

func processLeavesArch(input []byte, cvs *[256]byte) bool {
	processLeavesARM64(unsafe.SliceData(input), &cvs[0])
	return true
}

// processLeavesPairArch computes 2 leaf CVs from 2 contiguous chunks via a
// single x2 NEON pair, reading directly from the input with no scratch buffer.
func processLeavesPairArch(input []byte, cvs *[256]byte) bool {
	processLeavesPairARM64(unsafe.SliceData(input), &cvs[0])
	return true
}

// processLeavesRunArch reports that no multi-leaf run kernel is used on arm64;
// the 2-wide pair pass already drains remainders to fewer than two leaves.
func processLeavesRunArch(_ []byte, _ int, _ *[256]byte) bool { return false }

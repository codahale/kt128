//go:build arm64 && !purego

package kt128

import "unsafe"

const availableLanes = 8

//go:noescape
func processLeavesARM64(input *byte, cvs *byte)

//go:noescape
func processLeavesPairARM64(input *byte, cvs *byte)

//go:noescape
func processS0LeafPairARM64(input *byte, state *uint64, cv *byte)

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

// processS0LeafPairArch fuses the final node's absorption of S_0 || kt12
// marker with the first leaf's compression in one x2 NEON pass. input must be
// 2*BlockSize contiguous bytes (S_0 then the first leaf) and final must be a
// zero sponge; on return final holds the state after S_0 || marker and cv the
// leaf's chain value.
func processS0LeafPairArch(input []byte, final *sponge, cv *[32]byte) bool {
	processS0LeafPairARM64(unsafe.SliceData(input), &final.a[0], &cv[0])
	final.pos = BlockSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	return true
}

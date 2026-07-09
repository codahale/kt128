//go:build arm64 && !purego

package kt128

import "unsafe"

const availableLanes = 8

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss: the x2 pair kernel runs within ~5% of
// the x8 kernel per byte, so any even count is fine.
const flushChunks = 2

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

// processS0LeavesArch fuses the final node's absorption of S_0 || kt12 marker
// with leaf compression in one x2 NEON pass. arm64 supports exactly n == 2:
// input must be 2*BlockSize contiguous bytes (S_0 then the first leaf) and
// final must be a zero sponge. On return, final holds the state after
// S_0 || marker and cvs[32:64] the leaf's chain value.
func processS0LeavesArch(input []byte, n int, final *sponge, cvs *[256]byte) bool {
	if n != 2 {
		return false
	}
	processS0LeafPairARM64(unsafe.SliceData(input), &final.a[0], &cvs[32])
	final.pos = BlockSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	return true
}

// fuseS0Chunks returns how many chunks (S_0 plus leaves) the fused kernel
// should consume from a first write containing the given number of full
// chunks, or 0 to skip fusion. The x2 pair kernel always takes two; fusion is
// skipped when the leaves after S_0 form whole SIMD-width batches, since
// consuming one would strand lanes-1 of them in the buffer instead of
// flushing them all directly (measured +2.4% and an 8 KiB allocation at
// 72 KiB).
func fuseS0Chunks(chunks int) int {
	leaves := chunks - 1
	if leaves >= 1 && (leaves < availableLanes || leaves%availableLanes != 0) {
		return 2
	}
	return 0
}

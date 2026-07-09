//go:build arm64 && !purego

package kt128

import "unsafe"

const availableLanes = 8

// streamChunks is the streaming-path flush unit: two 5-chunk hybrid batches,
// so buffered flushes ride the hybrid kernel instead of parity-reducing to
// the pure-NEON x8 path.
const streamChunks = 10

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss: the x2 pair kernel runs within ~5% of
// the x8 kernel per byte, so any even count is fine.
const flushChunks = 2

// pairRemainderMax bounds the leaf counts the pair loop may drain; the pair
// kernel is the fastest narrow option at any remainder on arm64.
const pairRemainderMax = availableLanes

// hasLeafBatch5 reports that this platform can drain complete leaves in
// 5-chunk hybrid scalar/NEON batches.
const hasLeafBatch5 = true

//go:noescape
func processLeavesARM64(input *byte, cvs *byte)

//go:noescape
func processLeaves5ARM64(input *byte, cvs *byte)

//go:noescape
func processLeavesPairARM64(input *byte, cvs *byte)

//go:noescape
func processS0LeafPairARM64(input *byte, state *uint64, cv *byte)

//go:noescape
func processLeafPairPartialARM64(in0, in1 *byte, nShared uint64, cv *byte, lane1 *uint64)

// fuseTailChunks returns how many trailing complete leaves finalization
// should fold into one pass with the partial leaf's whole rate-blocks, or 0
// to keep the serial path. The arm64 pair kernel hosts exactly one complete
// leaf, and pairing pays only where the batch would otherwise strand it in a
// serial x1 pass: counts 1 and 3 (every other count drains through the
// batch5, x8, and pair kernels).
func fuseTailChunks(nFull, _ int) int {
	if nFull == 1 || nFull == 3 {
		return 1
	}
	return 0
}

// processLeavesTailArch computes the trailing complete leaf's CV while
// absorbing the partial leaf head's nShared whole rate-blocks into partial's
// state in the same 2-wide pass; the caller finishes the partial leaf's
// ragged tail and padding through the sponge. trailing must hold the
// complete chunk followed contiguously by the partial head; the pair kernel
// hosts exactly one complete leaf (n == 1).
func processLeavesTailArch(trailing []byte, n, nShared int, cvs *[256]byte, partial *sponge) bool {
	if n != 1 {
		return false
	}
	processLeafPairPartialARM64(unsafe.SliceData(trailing), unsafe.SliceData(trailing[BlockSize:]), uint64(nShared), &cvs[0], &partial.a[0])
	return true
}

func processLeavesArch(input []byte, cvs *[256]byte) bool {
	processLeavesARM64(unsafe.SliceData(input), &cvs[0])
	return true
}

// processLeavesBatch5Arch computes 5 leaf CVs from 5 contiguous chunks via the
// hybrid scalar/NEON kernel: chunks 0-3 as two x2 NEON pair passes and chunk 4
// on the scalar pipes, woven into the NEON round stream. Input must be
// 5*BlockSize contiguous bytes; the CVs land in cvs[:160].
func processLeavesBatch5Arch(input []byte, cvs *[256]byte) bool {
	processLeaves5ARM64(unsafe.SliceData(input), &cvs[0])
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
func fuseS0Chunks(chunks, _ int) int {
	leaves := chunks - 1
	if leaves >= 1 && (leaves < availableLanes || leaves%availableLanes != 0) {
		return 2
	}
	return 0
}

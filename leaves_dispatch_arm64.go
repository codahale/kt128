//go:build arm64 && !purego

package kt128

import "unsafe"

// ─── Scheduling policy ───
//
// arm64 leans on narrow NEON kernels: the bulk path is the 5-chunk hybrid
// scalar/NEON batch (four leaves at 2-wide NEON throughput plus a fifth on
// the scalar pipes), with an x3 hybrid and x2 pairs draining remainders. There
// is no dedicated x8 kernel — it was four sequential pair passes in one call.

const availableLanes = 8

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss: the x2 pair kernel runs within ~5% of
// the batch kernels per byte, so any even count is fine.
func flushChunks() int { return 2 }

// directFlushChunks returns the complete-chunk prefix to process from a direct
// write. Odd counts ending in 3, 5, 7, or 9 use hybrid batches plus pairs more
// cheaply than flushing the even prefix and eventually processing the stranded
// leaf serially. Counts ending in one retain that leaf so a later write can
// complete a faster batch.
func directFlushChunks(n int) int {
	switch n % 10 {
	case 3, 5, 7, 9:
		return n
	default:
		return n &^ 1
	}
}

// streamChunks is the streaming-path flush unit: one 5-chunk hybrid batch,
// so buffered flushes ride the hybrid kernel instead of parity-reducing to
// pure-NEON pairs. A single batch flushes sooner than two — a 10-chunk unit
// strands sub-10 messages in the buffer until finalization's pair-only drain
// (measured +11.5% at 64 KiB streaming) — and caps the buffer at one batch.
const streamChunks = 5

// growJumpMin is the buffered byte count at which a regrowing leaf buffer
// jumps straight to the streaming high-water mark instead of letting append
// re-copy through its doubling steps. On arm64 any regrowth jumps: the
// 48 KiB high-water allocation is cheap here, and eager jumping measured
// faster at every streaming size (up to -10% at 64 KiB, M4 Pro).
const growJumpMin = 0

// hasLeafX8 reports that arm64 has no dedicated x8 kernel; a remainder of
// eight drains through the pair loop at the same cost.
const hasLeafX8 = false

// hasLeafBatch5 reports that this platform can drain complete leaves in
// 5-chunk hybrid scalar/NEON batches.
const hasLeafBatch5 = true

// pairRemainderMax bounds the leaf counts the pair loop may drain after the x3
// hybrid has handled a three-leaf remainder.
const pairRemainderMax = availableLanes

// fuseS0Chunks returns how many chunks (S_0 plus leaves) the fused kernel
// should consume from a first write containing the given number of full
// chunks, or 0 to skip fusion. Three nearly chunk-aligned inputs use the x3
// hybrid for S_0 plus two leaves; other fused passes use the x2 pair. Fusion
// is skipped when the leaves after S_0 form whole SIMD-width batches, since
// consuming one would strand lanes-1 of them in the buffer instead of
// flushing them all directly (measured +2.4% and an 8 KiB allocation at
// 72 KiB).
func fuseS0Chunks(chunks, tail int) int {
	if chunks == 3 && tail < tripleSerialTailBlocks*rate {
		return 3
	}
	leaves := chunks - 1
	if leaves >= 1 && (leaves < availableLanes || leaves%availableLanes != 0) {
		return 2
	}
	return 0
}

// fuseTailChunks returns how many trailing complete leaves finalization
// should fold into one pass with the partial leaf's whole rate-blocks, or 0
// to keep the serial path. The arm64 pair kernel hosts exactly one complete
// leaf. A single complete leaf always pairs with the tail. Three complete
// leaves use x3 plus a serial short tail, switching to pair-plus-pair only once
// the tail is long enough to hide meaningful work in the second pair lane.
func fuseTailChunks(nFull, nShared int) int {
	if nFull == 1 || (nFull == 3 && nShared >= tripleSerialTailBlocks) {
		return 1
	}
	return 0
}

// tripleSerialTailBlocks is the partial-leaf length where an x3 hybrid plus
// serial x1 tail crosses the cost of two pair passes. Below 32 rate blocks the
// x3 route wins; at and above it, the complete leaf and tail share a pair.
const tripleSerialTailBlocks = 32

// ─── Kernels ───

// ─── Kernel wrappers ───

// processLeavesArch reports that arm64 has no x8 kernel (see hasLeafX8).
func processLeavesArch(_ []byte, _ *[256]byte) bool { return false }

// processLeavesBatch5Arch computes 5 leaf CVs from 5 contiguous chunks via the
// hybrid scalar/NEON kernel: chunks 0-3 as two x2 NEON pair passes and chunk 4
// on the scalar pipes, woven into the NEON round stream. Input must be
// 5*BlockSize contiguous bytes; the CVs land in cvs[:160].
func processLeavesBatch5Arch(input []byte, cvs *[256]byte) bool {
	processLeaves5ARM64(unsafe.SliceData(input), &cvs[0])
	return true
}

// processLeavesTripleArch computes 3 leaf CVs by weaving one scalar leaf into
// a 2-wide NEON pair and finishing its remaining half after the pair closes.
func processLeavesTripleArch(input []byte, cvs *[256]byte) bool {
	var scalar sponge
	processLeaves3ARM64(unsafe.SliceData(input), unsafe.SliceData(input[2*BlockSize:]), &cvs[0], &scalar.a[0])
	scalar.absorbAll(input[2*BlockSize+25*rate:], leafDS)
	scalar.squeeze(cvs[64:96])
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
// with leaf compression. Two chunks use the x2 NEON pair; three use the x3
// hybrid with S_0 on the scalar lane. final must be a zero sponge.
func processS0LeavesArch(input []byte, n int, final *sponge, cvs *[256]byte) bool {
	if n == 3 {
		processLeaves3ARM64(unsafe.SliceData(input[BlockSize:]), unsafe.SliceData(input), &cvs[32], &final.a[0])
		final.absorb(input[25*rate : BlockSize])
		final.absorb(kt12Marker[:])
		return true
	}
	if n != 2 {
		return false
	}
	processS0LeafPairARM64(unsafe.SliceData(input), &final.a[0], &cvs[32])
	final.pos = BlockSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	return true
}

// processS0LeavesTailArch reports that arm64 has no S_0+leaves+partial fused
// kernel; the 2-wide S_0 pair is always full.
func processS0LeavesTailArch(_ []byte, _, _ int, _, _ *sponge, _ *[256]byte) bool { return false }

// fuseS0TailBlocks reports that no partial-chunk blocks ride the fused S_0
// pass on arm64 (see processS0LeavesTailArch).
func fuseS0TailBlocks(_, _ int) int { return 0 }

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

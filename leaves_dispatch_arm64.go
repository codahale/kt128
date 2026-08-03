//go:build arm64 && !purego

package kt128

import (
	"unsafe"

	"github.com/codahale/kt128/internal/cpuid"
)

// ─── Scheduling policy ───
//
// arm64 leans on narrow NEON kernels: the bulk path is the 5-chunk hybrid
// scalar/NEON batch (four leaves at 2-wide NEON throughput plus a fifth on
// the scalar pipes), with an x3 hybrid and x2 pairs draining remainders. There
// is no dedicated x8 kernel — it was four sequential pair passes in one call.

const availableLanes = 8

// pairRemainderMax bounds the leaf counts the pair loop may drain after the x3
// hybrid has handled a three-leaf remainder.
const pairRemainderMax = availableLanes

func recommendedWriteBufferChunks() int {
	if cpuid.HasSHA3 {
		return 5
	}
	return 1
}

// fuseS0Chunks returns how many chunks (S_0 plus leaves) the fused kernel
// should consume from a first write containing the given number of full
// chunks, or 0 to skip fusion. Three nearly chunk-aligned inputs use the x3
// hybrid for S_0 plus two leaves; other fused passes use the x2 pair. Fusion
// is skipped when the leaves after S_0 form whole SIMD-width batches, since
// consuming one would turn a favorable whole-batch shape into a narrow
// remainder.
func fuseS0Chunks(chunks, tail int) int {
	if !cpuid.HasSHA3 {
		return 0
	}
	if chunks == 3 && tail < tripleSerialTailBlocks*rate {
		return 3
	}
	leaves := chunks - 1
	if leaves >= 1 && (leaves < availableLanes || leaves%availableLanes != 0) {
		return 2
	}
	return 0
}

// fuseTailChunks returns how many trailing complete leaves to fold into one
// pass with a partial leaf's whole rate-blocks, or 0 to keep the serial path.
// The arm64 pair kernel hosts exactly one complete
// leaf. A single complete leaf always pairs with the tail. Three complete
// leaves use x3 plus a serial short tail, switching to pair-plus-pair only once
// the tail is long enough to hide meaningful work in the second pair lane.
func fuseTailChunks(nFull, nShared int) int {
	if !cpuid.HasSHA3 {
		return 0
	}
	if nFull == 1 || (nFull == 3 && nShared >= tripleSerialTailBlocks) {
		return 1
	}
	return 0
}

// tripleSerialTailBlocks is the partial-leaf length where an x3 hybrid plus
// serial x1 tail crosses the cost of two pair passes. Below 32 rate blocks the
// x3 route wins; at and above it, the complete leaf and tail share a pair.
const tripleSerialTailBlocks = 32

// ─── Kernel wrappers ───

// Every try wrapper returns false without modifying its output arguments when
// the requested kernel is unavailable or does not support the requested shape.

// tryProcessLeavesX8Arch reports that arm64 has no x8 kernel; eight leaves
// drain through the pair loop at the same cost.
func tryProcessLeavesX8Arch(_ []byte, _ *[256]byte) bool { return false }

// tryProcessLeavesBatch5Arch computes 5 leaf CVs from 5 contiguous chunks via
// the hybrid scalar/NEON kernel: chunks 0-3 as two x2 NEON pair passes and
// chunk 4 on the scalar pipes, woven into the NEON round stream. Input must be
// 5*ChunkSize contiguous bytes; the CVs land in cvs[:160].
func tryProcessLeavesBatch5Arch(input []byte, cvs *[256]byte) bool {
	if !cpuid.HasSHA3 {
		return false
	}
	processLeaves5ARM64(unsafe.SliceData(input), &cvs[0])
	return true
}

// tryProcessLeavesTripleArch computes 3 leaf CVs by weaving one scalar leaf into
// a 2-wide NEON pair and finishing its remaining half after the pair closes.
func tryProcessLeavesTripleArch(input []byte, cvs *[256]byte) bool {
	if !cpuid.HasSHA3 {
		return false
	}
	var scalar sponge
	processLeaves3ARM64(unsafe.SliceData(input), unsafe.SliceData(input[2*ChunkSize:]), &cvs[0], &scalar.a[0])
	scalar.absorbAll(input[2*ChunkSize+25*rate:], leafDS)
	scalar.squeeze(cvs[64:96])
	return true
}

// tryProcessLeavesPairArch computes 2 leaf CVs from 2 contiguous chunks via a
// single x2 NEON pair, reading directly from the input with no scratch buffer.
func tryProcessLeavesPairArch(input []byte, cvs *[256]byte) bool {
	if !cpuid.HasSHA3 {
		return false
	}
	processLeavesPairARM64(unsafe.SliceData(input), &cvs[0])
	return true
}

// tryProcessLeavesRunArch reports that no multi-leaf run kernel is used on arm64;
// the 2-wide pair pass already drains remainders to fewer than two leaves.
func tryProcessLeavesRunArch(_ []byte, _ int, _ *[256]byte) bool { return false }

// tryProcessS0LeavesArch fuses the final node's absorption of S_0 || kt12 marker
// with leaf compression. Two chunks use the x2 NEON pair; three use the x3
// hybrid with S_0 on the scalar lane. final must be a zero sponge.
func tryProcessS0LeavesArch(input []byte, n int, final *sponge, cvs *[256]byte) bool {
	if !cpuid.HasSHA3 {
		return false
	}
	if n == 3 {
		processLeaves3ARM64(unsafe.SliceData(input[ChunkSize:]), unsafe.SliceData(input), &cvs[32], &final.a[0])
		final.absorb(input[25*rate : ChunkSize])
		final.absorb(kt12Marker[:])
		return true
	}
	if n != 2 {
		return false
	}
	processS0LeafPairARM64(unsafe.SliceData(input), &final.a[0], &cvs[32])
	final.pos = ChunkSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	return true
}

// tryProcessS0LeavesTailArch reports that arm64 has no S_0+leaves+partial fused
// kernel; the 2-wide S_0 pair is always full.
func tryProcessS0LeavesTailArch(_ []byte, _, _ int, _, _ *sponge, _ *[256]byte) bool {
	return false
}

// fuseS0TailBlocks reports that no partial-chunk blocks ride the fused S_0
// pass on arm64 (see tryProcessS0LeavesTailArch).
func fuseS0TailBlocks(_, _ int) int { return 0 }

// tryProcessLeavesTailArch computes the trailing complete leaf's CV while
// absorbing the partial leaf head's nShared whole rate-blocks into partial's
// state in the same 2-wide pass; the caller finishes the partial leaf's
// ragged tail and padding through the sponge. trailing must hold the
// complete chunk followed contiguously by the partial head; the pair kernel
// hosts exactly one complete leaf (n == 1).
func tryProcessLeavesTailArch(trailing []byte, n, nShared int, cvs *[256]byte, partial *sponge) bool {
	if !cpuid.HasSHA3 || n != 1 {
		return false
	}
	processLeafPairPartialARM64(unsafe.SliceData(trailing), unsafe.SliceData(trailing[ChunkSize:]), uint64(nShared), &cvs[0], &partial.a[0])
	return true
}

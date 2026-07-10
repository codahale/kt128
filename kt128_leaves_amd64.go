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
// remainder paths at 4-wide, S_0 and tail fusion ride quad variants (up to
// four chunks per fused pass), and the pair kernel reports unavailable —
// without VPROLQ/VPTERNLOGQ a narrow pass cannot beat the quad's flat cost.

const availableLanes = 8

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss. On AVX-512 it is the full SIMD width:
// the remainder paths are masked passes whose cost is flat at any lane
// occupancy. On AVX2 it is one quad: the quad kernel's cost is flat at four
// lanes, so a quad-sized tail flushes directly instead of riding the buffer.
func flushChunks() int {
	if cpuid.HasAVX512 {
		return availableLanes
	}
	return 4
}

// streamChunks is the streaming-path flush unit; amd64 has no hybrid batch
// kernel, so it is the SIMD width.
const streamChunks = availableLanes

// growJumpMin is the buffered byte count at which a regrowing leaf buffer
// jumps straight to the streaming high-water mark instead of letting append
// re-copy through its doubling steps. Jumping eagerly on any regrowth
// measured -17% at 64 KiB streaming but +5..7% at 28-32 KiB (Emerald
// Rapids): short streams never fill the 72 KiB high-water allocation and
// only pay for zeroing it. Four chunks is past every shape that stops short
// (a 32 KiB stream peaks at three buffered chunks) while keeping most of
// the large-stream win (-4.7% at 64 KiB).
const growJumpMin = 4 * BlockSize

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
	if chunks < 2 {
		return 0
	}
	if cpuid.HasAVX512 {
		if chunks <= availableLanes || chunks%availableLanes != 1 || tail >= rate {
			return min(chunks, availableLanes)
		}
		return 0
	}
	// AVX2: the quad hosts S_0 plus up to three leaves, saving the serial
	// S_0 pass. Exactly two chunks always fuse: there are no leftovers, and
	// the quad beats the serial S_0-plus-x1 pair it replaces (measured +11.7%
	// at 16 KiB, Emerald Rapids). Above two, the exceptions fall where
	// consuming four chunks worsens the remaining count's drain enough to
	// cancel the saving: counts of 1 mod 8 (fusing turns a clean
	// multiple-of-8 drain into a 5-mod-8 remainder), 5 mod 8 (the fused
	// leftovers strand a leaf), and 2 mod 8, where a quad costs about the two
	// serial passes fusion saves but the fused leftovers add a net remainder
	// quad plus buffering — a measured net loss.
	if chunks == 2 {
		return 2
	}
	if r := chunks % availableLanes; r == 1 || r == 2 || r == 5 {
		return 0
	}
	return min(chunks, 4)
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
	if nShared == 0 {
		return 0
	}
	if cpuid.HasAVX512 {
		return nFull % 8
	}
	// AVX2: the tail rides a 1..3-leaf remainder in a quad; a leading
	// multiple of four drains through the batch and quad run kernels. A bare
	// single trailing leaf stays serial: with no narrow kernel the tail
	// would ride a full quad, which costs about two serial passes — more
	// than the x1 plus at-most-one-pass serial tail it replaces. (At five
	// leaves the n=1 quad instead replaces the run kernel's second
	// remainder quad, so it wins and stays fused.)
	if nFull == 1 {
		return 0
	}
	return nFull % 4
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

//go:noescape
func processS0LeavesTailAVX512(input *byte, state *uint64, cvs *byte, n, nShared uint64, tail *uint64)

//go:noescape
func processS0LeavesQuadAVX2(input *byte, state *uint64, cvs *byte, n uint64)

//go:noescape
func processLeavesQuadTailAVX2(input *byte, cvs *byte, n, nShared uint64, lane1 *uint64)

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
// with leaf compression in one pass: on AVX-512 a 2-wide XMM pair at n == 2
// (pair cost instead of a flat masked pass) or an 8-wide pass for n in 3..8;
// on AVX2 a quad pass for n in 2..4. input must be n*BlockSize contiguous
// bytes (S_0 then n-1 leaves) and final must be a zero sponge. On return,
// final holds the state after S_0 || marker and cvs[32:n*32] the leaves'
// chain values.
func processS0LeavesArch(input []byte, n int, final *sponge, cvs *[256]byte) bool {
	switch {
	case n < 2:
		return false
	case !cpuid.HasAVX512:
		if n > 4 {
			return false
		}
		processS0LeavesQuadAVX2(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n))
	case n == 2:
		processS0LeafPairAVX512(unsafe.SliceData(input), &final.a[0], &cvs[32])
	case n <= 8:
		processS0LeavesAVX512(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n))
	default:
		return false
	}
	final.pos = BlockSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	return true
}

// processS0LeavesTailArch fuses processS0LeavesArch with the trailing partial
// leaf: one masked 8-wide pass computes the final node's S_0 || marker state
// (lane 0), n-1 leaf CVs (cvs[32:n*32]), and absorbs the partial leaf's
// nShared whole rate-blocks — read from input[n*BlockSize:] — into lane n,
// whose mid-absorption state lands in pending for the caller to continue.
// input must be at least n*BlockSize + nShared*rate contiguous bytes and
// final a zero sponge. AVX-512 only; n must be in 2..7 so lane n is free.
func processS0LeavesTailArch(input []byte, n, nShared int, final, pending *sponge, cvs *[256]byte) bool {
	if !cpuid.HasAVX512 || n < 2 || n > 7 {
		return false
	}
	processS0LeavesTailAVX512(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n), uint64(nShared), &pending.a[0])
	final.pos = BlockSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	pending.pos = 0                              // whole rate-blocks absorbed
	return true
}

// fuseS0TailBlocks returns how many whole rate-blocks of a first write's
// trailing partial chunk should ride the fused S_0 pass in an otherwise-idle
// lane, or 0 to leave the tail for finalization. The caller must be fusing
// every complete chunk (the partial's data directly follows them). On
// AVX-512 with 2..7 chunks the pass has a free lane and its masked cost is
// flat, so the tail's blocks ride free — except at exactly two chunks, where
// S_0 fusion otherwise takes a cheaper 2-wide XMM pair: the flat masked pass
// pays off only once the tail is big enough that the serial pass it replaces
// costs more than the pair saves (measured breakeven near half a chunk,
// Emerald Rapids).
func fuseS0TailBlocks(chunks, tail int) int {
	if !cpuid.HasAVX512 || chunks < 2 || chunks > 7 {
		return 0
	}
	n := tail / rate
	if chunks == 2 && n < s0TailPairMin {
		return 0
	}
	return n
}

// s0TailPairMin is the tail size, in whole rate-blocks, at which a two-chunk
// first write switches from the 2-wide XMM S_0 pair (plus a serial tail at
// finalization) to the flat masked pass with the tail riding a third lane:
// the pair path costs ~7.1µs plus ~135ns per serial tail block and the
// masked pass a flat ~9.8µs (measured Emerald Rapids), crossing at 27.
const s0TailPairMin = 27

// processLeavesTailArch computes n trailing complete leaf CVs while
// absorbing the following partial leaf's nShared whole rate-blocks into
// partial's state in the same pass — on AVX-512 a 2-wide XMM pair for n == 1
// or a masked gather pass for n in 2..7, on AVX2 a quad pass with a tail
// lane for n in 1..3; the caller finishes the partial leaf's ragged tail and
// padding through the sponge. trailing must hold the n complete chunks
// followed contiguously by the partial head.
func processLeavesTailArch(trailing []byte, n, nShared int, cvs *[256]byte, partial *sponge) bool {
	if !cpuid.HasAVX512 {
		if n < 1 || n > 3 {
			return false
		}
		processLeavesQuadTailAVX2(unsafe.SliceData(trailing), &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
		return true
	}
	if n == 1 {
		processLeafPairPartialAVX512(unsafe.SliceData(trailing), unsafe.SliceData(trailing[BlockSize:]), uint64(nShared), &cvs[0], &partial.a[0])
		return true
	}
	processLeavesRunPartialAVX512(unsafe.SliceData(trailing), &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
	return true
}

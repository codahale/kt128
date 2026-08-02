//go:build amd64 && !purego

package kt128

import (
	"unsafe"

	"github.com/codahale/kt128/internal/cpuid"
)

// ─── Scheduling policy ───
//
// amd64 matches kernel width to live lanes: whole 8-leaf batches run as
// 8-wide AVX-512 passes, 3..4-lane shapes as 4-wide YMM quads (256-bit ops
// issue on three vector ALU ports where 512-bit get two, so a quad pass
// costs ~0.70 of a masked 8-wide pass), 5..7-leaf remainders as flat masked
// 8-wide passes, and 2-lane shapes as 2-wide XMM pairs (equal pass cost to
// the quad, no masking setup; XMM and YMM passes measure identical, so
// nothing narrower than the quad is ever cheaper). Without AVX-512, when AVX2
// is available, the x8 and quad kernels cover the batch and remainder paths at
// 4-wide, S_0 and tail fusion ride quad variants (up to four chunks per fused
// pass), and the pair kernel reports unavailable — without VPROLQ/VPTERNLOGQ a
// narrow pass cannot beat the quad's flat cost. CPUs without either ISA use the
// generic single-leaf path.

const availableLanes = 8

// pairRemainderMax bounds the leaf counts the pair loop may drain: a 2-wide
// XMM pass costs ~0.63 of a flat masked pass (Emerald Rapids), so a pair
// wins for a remainder of exactly two but chains of pairs lose to the
// masked run kernel from three up.
const pairRemainderMax = 2

func recommendedWriteBufferChunks() int {
	if cpuid.HasAVX512 || cpuid.HasAVX2 {
		return availableLanes
	}
	return 1
}

// fuseS0Chunks returns how many chunks (S_0 plus leaves) the fused kernel
// should consume from a first write containing the given number of full
// chunks (plus tail trailing bytes), or 0 to skip fusion. Up to
// availableLanes chunks fuse into one pass (2-wide XMM at exactly two,
// 8-wide masked above), replacing the serial S_0 absorption: at or below one
// pass this is a strict win, and above it the 1..7 chunks the fused pass
// strands past the last whole batch drain in one pass with the partial tail
// (tail fusion), so fusing still saves the serial S_0 pass outright. The
// exception is a chunk count of 1 mod availableLanes with less than a whole
// rate-block of tail: the stranded leaf then has no partial to pair with,
// its serial pass cancels the saving.
func fuseS0Chunks(chunks, tail int) int {
	if chunks < 2 {
		return 0
	}
	if cpuid.HasAVX512 {
		// Past one whole batch, counts of 2 and 4 mod 8 fuse narrow instead:
		// the XMM S_0 pair (~5.7µs) or YMM S_0 quad (~6.3µs) lands the drain
		// on a batch boundary, displacing a fused 8-wide pass plus a trailing
		// pair or quad (a small win exact — ~0.2-0.4µs — and a larger one
		// ragged, where the trailing chunks would otherwise drag the partial
		// into a 5-plus-lane masked finalize pass).
		if chunks > availableLanes {
			switch chunks % availableLanes {
			case 2:
				return 2
			case 4:
				return 4
			}
		}
		if chunks <= availableLanes || chunks%availableLanes != 1 || tail >= rate {
			return min(chunks, availableLanes)
		}
		return 0
	}
	if !cpuid.HasAVX2 {
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
	// quad plus a narrow remainder — a measured net loss.
	if chunks == 2 {
		return 2
	}
	if r := chunks % availableLanes; r == 1 || r == 2 || r == 5 {
		return 0
	}
	return min(chunks, 4)
}

// fuseTailChunks returns how many trailing complete leaves to fold into one
// pass with a partial leaf's whole rate-blocks, or 0 to keep the serial path.
// On AVX-512 the tail rides a 2..7-leaf remainder
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
	if !cpuid.HasAVX2 {
		return 0
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

// ─── Kernel wrappers ───

// Every try wrapper returns false without modifying its output arguments when
// the requested kernel is unavailable or does not support the requested shape.

func tryProcessLeavesX8Arch(input []byte, cvs *[256]byte) bool {
	if cpuid.HasAVX512 {
		processLeavesAVX512(unsafe.SliceData(input), &cvs[0])
		return true
	}
	if cpuid.HasAVX2 {
		processLeavesAVX2(unsafe.SliceData(input), &cvs[0])
		return true
	}
	return false
}

func tryProcessLeavesBatch5Arch(_ []byte, _ *[256]byte) bool { return false }

func tryProcessLeavesTripleArch(_ []byte, _ *[256]byte) bool { return false }

// tryProcessLeavesPairArch computes 2 leaf CVs from 2 contiguous chunks via a
// single 2-wide XMM pass, reading directly from the input with plain loads.
func tryProcessLeavesPairArch(input []byte, cvs *[256]byte) bool {
	if !cpuid.HasAVX512 {
		return false
	}
	processLeavesPairAVX512(unsafe.SliceData(input), &cvs[0])
	return true
}

// tryProcessLeavesRunArch computes n (2..7) leaf CVs by reading the chunks
// directly with no scratch buffer: on AVX-512 a 4-wide YMM quad pass for n up to 4
// (256-bit ops get three vector ALU ports where 512-bit get two, so the quad
// costs ~0.70 of the masked 8-wide pass) and the 8-wide masked-gather pass
// above that; on AVX2 one to two x4 passes with dummy lanes.
func tryProcessLeavesRunArch(data []byte, n int, cvs *[256]byte) bool {
	if n < 2 || n > 7 {
		return false
	}
	if cpuid.HasAVX512 {
		if n <= 4 {
			processLeavesQuadAVX512(unsafe.SliceData(data), &cvs[0], uint64(n))
		} else {
			processLeavesRunAVX512(unsafe.SliceData(data), &cvs[0], uint64(n))
		}
		return true
	}
	if !cpuid.HasAVX2 {
		return false
	}

	// AVX2: run x4 passes, pointing dummy lanes at an in-bounds chunk. The first
	// pass covers leaves 0..3, the second (only when n > 4) covers 4..n-1. CVs
	// for dummy lanes are written but never read by the caller.
	base := unsafe.Pointer(unsafe.SliceData(data))
	var p [8]*byte
	for i := range p {
		off := 0
		if i < n {
			off = i * ChunkSize // dummy lanes (i >= n) read chunk 0, discarded
		}
		p[i] = (*byte)(unsafe.Add(base, off))
	}
	processLeavesQuadAVX2(p[0], p[1], p[2], p[3], &cvs[0])
	if n > 4 {
		processLeavesQuadAVX2(p[4], p[5], p[6], p[7], &cvs[128])
	}
	return true
}

// tryProcessS0LeavesArch fuses the final node's absorption of S_0 || kt12 marker
// with leaf compression in one pass: on AVX-512 a 2-wide XMM pair at n == 2
// (pair cost instead of a quad's masking setup), a 4-wide YMM quad for n in
// 3..4, or an 8-wide pass for n in 5..8; on AVX2 a quad pass for n in 2..4.
// input must be n*ChunkSize contiguous bytes (S_0 then n-1 leaves) and final
// must be a zero sponge. On return, final holds the state after S_0 || marker
// and cvs[32:n*32] the leaves' chain values.
func tryProcessS0LeavesArch(input []byte, n int, final *sponge, cvs *[256]byte) bool {
	switch {
	case n < 2:
		return false
	case cpuid.HasAVX512:
		switch {
		case n == 2:
			processS0LeafPairAVX512(unsafe.SliceData(input), &final.a[0], &cvs[32])
		case n <= 4:
			processS0LeavesQuadAVX512(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n))
		case n <= 8:
			processS0LeavesAVX512(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n))
		default:
			return false
		}
	case cpuid.HasAVX2:
		if n > 4 {
			return false
		}
		processS0LeavesQuadAVX2(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n))
	default:
		return false
	}
	final.pos = ChunkSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	return true
}

// tryProcessS0LeavesTailArch fuses tryProcessS0LeavesArch with the trailing partial
// leaf: one pass computes the final node's S_0 || marker state (lane 0), n-1
// leaf CVs (cvs[32:n*32]), and absorbs the partial leaf's nShared whole
// rate-blocks — read from input[n*ChunkSize:] — into lane n, whose
// mid-absorption state lands in partial for the caller to continue. input
// must be at least n*ChunkSize + nShared*rate contiguous bytes and final a
// zero sponge. Lane n must be free: on AVX-512 a 4-wide YMM quad hosts n in
// 2..3 and a masked 8-wide pass n in 4..7; a quad pass hosts n in 2..3 on
// AVX2.
func tryProcessS0LeavesTailArch(input []byte, n, nShared int, final, partial *sponge, cvs *[256]byte) bool {
	switch {
	case n < 2:
		return false
	case cpuid.HasAVX512:
		if n <= 3 {
			processS0LeavesQuadTailAVX512(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
		} else if n <= 7 {
			processS0LeavesTailAVX512(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
		} else {
			return false
		}
	case cpuid.HasAVX2:
		if n > 3 {
			return false
		}
		processS0LeavesQuadTailAVX2(unsafe.SliceData(input), &final.a[0], &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
	default:
		return false
	}
	final.pos = ChunkSize%rate + len(kt12Marker) // mid-block after S_0 || marker
	partial.pos = 0                              // whole rate-blocks absorbed
	return true
}

// fuseS0TailBlocks returns how many whole rate-blocks of a first write's
// trailing partial chunk should ride the fused S_0 pass in an otherwise-idle
// lane, or 0 to leave the tail for ordinary incremental absorption. The caller
// must be fusing every complete chunk (the partial's data directly follows
// them). On
// AVX-512 with 2..7 chunks the pass has a free lane (a YMM quad up to three
// chunks, the masked 8-wide pass above), so the tail's blocks ride free —
// except at exactly two chunks, where S_0 fusion otherwise takes a cheaper
// 2-wide XMM pair: the quad with the tail riding a third lane pays off only
// once the tail is big enough that the serial pass it replaces costs more
// than the pair saves. On AVX2 the quad hosts S_0 plus up to two leaves with
// a lane to spare, and with no cheaper narrow kernel the S_0 fusion already
// rides the quad at two chunks, so the tail's blocks always ride free.
func fuseS0TailBlocks(chunks, tail int) int {
	n := tail / rate
	if cpuid.HasAVX512 {
		if chunks < 2 || chunks > 7 {
			return 0
		}
		if chunks == 2 && n < s0TailPairMin {
			return 0
		}
		return n
	}
	if cpuid.HasAVX2 {
		if chunks < 2 || chunks > 3 {
			return 0
		}
		return n
	}
	return 0
}

// s0TailPairMin is the tail size, in whole rate-blocks, at which a two-chunk
// first write switches from the 2-wide XMM S_0 pair (plus a serial tail at
// finalization) to the YMM quad with the tail riding a third lane: the pair
// path costs ~5.7µs plus ~106ns per serial tail block and the quad a flat
// ~6.5µs (measured Emerald Rapids), crossing at 7 blocks.
const s0TailPairMin = 7

// tryProcessLeavesTailArch computes n trailing complete leaf CVs while
// absorbing the following partial leaf's nShared whole rate-blocks into
// partial's state in the same pass — on AVX-512 a 2-wide XMM pair for n == 1,
// a 4-wide YMM quad for n in 2..3, or a masked gather pass for n in 4..7; on
// AVX2 a quad pass with a tail lane for n in 1..3. The caller finishes the
// partial leaf's ragged tail and padding through the sponge. trailing must
// hold the n complete chunks followed contiguously by the partial head.
func tryProcessLeavesTailArch(trailing []byte, n, nShared int, cvs *[256]byte, partial *sponge) bool {
	if n < 1 || n > 7 {
		return false
	}
	if cpuid.HasAVX512 {
		switch {
		case n == 1:
			processLeafPairPartialAVX512(unsafe.SliceData(trailing), unsafe.SliceData(trailing[ChunkSize:]), uint64(nShared), &cvs[0], &partial.a[0])
		case n <= 3:
			processLeavesQuadPartialAVX512(unsafe.SliceData(trailing), &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
		default:
			processLeavesRunPartialAVX512(unsafe.SliceData(trailing), &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
		}
		return true
	}
	if cpuid.HasAVX2 {
		if n < 1 || n > 3 {
			return false
		}
		processLeavesQuadTailAVX2(unsafe.SliceData(trailing), &cvs[0], uint64(n), uint64(nShared), &partial.a[0])
		return true
	}
	return false
}

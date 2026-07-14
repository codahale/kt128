//go:build amd64 && !purego

package kt128

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
func processLeavesQuadAVX512(input *byte, cvs *byte, n uint64)

//go:noescape
func processLeavesQuadPartialAVX512(input *byte, cvs *byte, n, nShared uint64, lane1 *uint64)

//go:noescape
func processS0LeavesQuadAVX512(input *byte, state *uint64, cvs *byte, n uint64)

//go:noescape
func processS0LeavesQuadTailAVX512(input *byte, state *uint64, cvs *byte, n, nShared uint64, tail *uint64)

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
func processS0LeavesQuadTailAVX2(input *byte, state *uint64, cvs *byte, n, nShared uint64, tail *uint64)

//go:noescape
func processLeavesQuadTailAVX2(input *byte, cvs *byte, n, nShared uint64, lane1 *uint64)

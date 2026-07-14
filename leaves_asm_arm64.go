//go:build arm64 && !purego

package kt128

//go:noescape
func processLeaves5ARM64(input *byte, cvs *byte)

//go:noescape
func processLeaves3ARM64(pairInput, scalarInput *byte, cvs *byte, state *uint64)

//go:noescape
func processLeavesPairARM64(input *byte, cvs *byte)

//go:noescape
func processS0LeafPairARM64(input *byte, state *uint64, cv *byte)

//go:noescape
func processLeafPairPartialARM64(in0, in1 *byte, nShared uint64, cv *byte, lane1 *uint64)

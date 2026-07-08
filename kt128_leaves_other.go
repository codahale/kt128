//go:build (!amd64 && !arm64) || purego

package kt128

const availableLanes = 1

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss; the scalar fallback has only one speed.
const flushChunks = 1

func processLeavesArch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesPairArch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesRunArch(_ []byte, _ int, _ *[256]byte) bool { return false }

func processS0LeafPairArch(_ []byte, _ *sponge, _ *[32]byte) bool { return false }

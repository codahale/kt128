//go:build (!amd64 && !arm64) || purego

package kt128

const availableLanes = 1

// streamChunks is the streaming-path flush unit; the scalar fallback has no
// batch kernel, so it is a single chunk.
const streamChunks = availableLanes

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss; the scalar fallback has only one speed.
const flushChunks = 1

const hasLeafBatch5 = false

func processLeavesArch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesBatch5Arch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesPairArch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesRunArch(_ []byte, _ int, _ *[256]byte) bool { return false }

func processS0LeavesArch(_ []byte, _ int, _ *sponge, _ *[256]byte) bool { return false }

func fuseS0Chunks(_ int) int { return 0 }

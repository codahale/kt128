//go:build (!amd64 && !arm64) || purego

package kt128

// ─── Scheduling policy ───
//
// The scalar fallback has no batch kernels: every leaf runs serially through
// the sponge, so all batch policies report unavailable and every remainder
// drains through the x1 loop.

const availableLanes = 1

// flushChunks is the smallest chunk count the direct fast path may flush
// without meaningful throughput loss; the scalar fallback has only one speed.
const flushChunks = 1

// streamChunks is the streaming-path flush unit; the scalar fallback has no
// batch kernel, so it is a single chunk.
const streamChunks = availableLanes

// hasLeafX8 reports that the scalar fallback has no batch kernel; the
// generic 8-wide path is eight serial sponges, no faster than the x1 loop.
const hasLeafX8 = false

const hasLeafBatch5 = false

// pairRemainderMax bounds the leaf counts the pair loop may drain; there is
// no pair kernel on this platform.
const pairRemainderMax = 0

func fuseS0Chunks(_, _ int) int { return 0 }

func fuseTailChunks(_, _ int) int { return 0 }

// ─── Kernel wrappers ───

func processLeavesArch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesBatch5Arch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesPairArch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesRunArch(_ []byte, _ int, _ *[256]byte) bool { return false }

func processS0LeavesArch(_ []byte, _ int, _ *sponge, _ *[256]byte) bool { return false }

func processLeavesTailArch(_ []byte, _, _ int, _ *[256]byte, _ *sponge) bool { return false }

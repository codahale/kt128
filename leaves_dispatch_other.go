//go:build (!amd64 && !arm64) || purego

package kt128

// ─── Scheduling policy ───
//
// The scalar fallback has no batch kernels: every leaf runs serially through
// the sponge, so all batch policies report unavailable and every remainder
// drains through the x1 loop.

const availableLanes = 1

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

func processLeavesTripleArch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesPairArch(_ []byte, _ *[256]byte) bool { return false }

func processLeavesRunArch(_ []byte, _ int, _ *[256]byte) bool { return false }

func processS0LeavesArch(_ []byte, _ int, _ *sponge, _ *[256]byte) bool { return false }

func processS0LeavesTailArch(_ []byte, _, _ int, _, _ *sponge, _ *[256]byte) bool { return false }

func fuseS0TailBlocks(_, _ int) int { return 0 }

func processLeavesTailArch(_ []byte, _, _ int, _ *[256]byte, _ *sponge) bool { return false }

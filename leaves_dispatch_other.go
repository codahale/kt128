//go:build (!amd64 && !arm64) || purego

package kt128

// ─── Scheduling policy ───
//
// The scalar fallback has no batch kernels: every leaf runs serially through
// the sponge, so all batch policies report unavailable and every remainder
// drains through the x1 loop.

const availableLanes = 1

// pairRemainderMax bounds the leaf counts the pair loop may drain; there is
// no pair kernel on this platform.
const pairRemainderMax = 0

func recommendedWriteBufferChunks() int { return 1 }

func fuseS0Chunks(_, _ int) int { return 0 }

func fuseTailChunks(_, _ int) int { return 0 }

// ─── Kernel wrappers ───

// Every try wrapper returns false without modifying its output arguments; the
// scalar fallback has no architecture-specific kernels.

func tryProcessLeavesX8Arch(_ []byte, _ *[256]byte) bool { return false }

func tryProcessLeavesBatch5Arch(_ []byte, _ *[256]byte) bool { return false }

func tryProcessLeavesTripleArch(_ []byte, _ *[256]byte) bool { return false }

func tryProcessLeavesPairArch(_ []byte, _ *[256]byte) bool { return false }

func tryProcessLeavesRunArch(_ []byte, _ int, _ *[256]byte) bool { return false }

func tryProcessS0LeavesArch(_ []byte, _ int, _ *sponge, _ *[256]byte) bool { return false }

func tryProcessS0LeavesTailArch(_ []byte, _, _ int, _, _ *sponge, _ *[256]byte) bool {
	return false
}

func fuseS0TailBlocks(_, _ int) int { return 0 }

func tryProcessLeavesTailArch(_ []byte, _, _ int, _ *[256]byte, _ *sponge) bool {
	return false
}

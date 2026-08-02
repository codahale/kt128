package kt128

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

type kernelTry func(*[256]byte, *sponge, *sponge) bool

func assertTryKernelFailure(t *testing.T, reason string, try kernelTry) {
	t.Helper()
	var cvs [256]byte
	for i := range cvs {
		cvs[i] = byte(i) ^ 0xA5
	}
	final := sponge{pos: 17}
	final.a[0] = 1
	partial := sponge{pos: 23}
	partial.a[0] = 2
	wantCVs, wantFinal, wantPartial := cvs, final, partial

	if try(&cvs, &final, &partial) {
		t.Fatalf("%s kernel reported success", reason)
	}
	if cvs != wantCVs || final != wantFinal || partial != wantPartial {
		t.Fatalf("%s kernel modified its outputs", reason)
	}
}

// testUnavailableKernelWrappers pins the try-wrapper contract after an
// architecture test has disabled its CPU features: false means no output was
// modified, so generic fallback remains safe even if scheduling and dispatch
// observe different capability state.
func testUnavailableKernelWrappers(t *testing.T) {
	t.Helper()
	input := make([]byte, 8*ChunkSize+rate)

	tests := []struct {
		name string
		try  kernelTry
	}{
		{"x8", func(cvs *[256]byte, _, _ *sponge) bool {
			return tryProcessLeavesX8Arch(input[:8*ChunkSize], cvs)
		}},
		{"batch5", func(cvs *[256]byte, _, _ *sponge) bool {
			return tryProcessLeavesBatch5Arch(input[:5*ChunkSize], cvs)
		}},
		{"triple", func(cvs *[256]byte, _, _ *sponge) bool {
			return tryProcessLeavesTripleArch(input[:3*ChunkSize], cvs)
		}},
		{"pair", func(cvs *[256]byte, _, _ *sponge) bool {
			return tryProcessLeavesPairArch(input[:2*ChunkSize], cvs)
		}},
		{"run", func(cvs *[256]byte, _, _ *sponge) bool {
			return tryProcessLeavesRunArch(input[:2*ChunkSize], 2, cvs)
		}},
		{"s0", func(cvs *[256]byte, final, _ *sponge) bool {
			return tryProcessS0LeavesArch(input[:2*ChunkSize], 2, final, cvs)
		}},
		{"s0_tail", func(cvs *[256]byte, final, partial *sponge) bool {
			return tryProcessS0LeavesTailArch(input[:2*ChunkSize+rate], 2, 1, final, partial, cvs)
		}},
		{"leaf_tail", func(cvs *[256]byte, _, partial *sponge) bool {
			return tryProcessLeavesTailArch(input[:ChunkSize+rate], 1, 1, cvs, partial)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assertTryKernelFailure(t, "unavailable", tc.try)
		})
	}

	t.Run("fused_leaf_caller", func(t *testing.T) {
		h := New(nil)
		h.final = sponge{pos: 17}
		h.final.a[0] = 1
		h.leaf = sponge{pos: 23}
		h.leaf.a[0] = 2
		h.leafLen = 31
		wantFinal, wantLeaf, wantLeafLen := h.final, h.leaf, h.leafLen

		if h.startLeafFused(input[:ChunkSize+rate], 1, input[ChunkSize:ChunkSize+rate]) {
			t.Fatal("unavailable fused leaf kernel reported success")
		}
		if h.final != wantFinal || h.leaf != wantLeaf || h.leafLen != wantLeafLen {
			t.Fatal("failed fused leaf dispatch modified the Hasher")
		}
	})
}

func TestTryKernelInvalidShapesDoNotMutate(t *testing.T) {
	input := make([]byte, 8*ChunkSize+rate)
	tests := []struct {
		name string
		try  kernelTry
	}{
		{"run_below_range", func(cvs *[256]byte, _, _ *sponge) bool {
			return tryProcessLeavesRunArch(input[:ChunkSize], 1, cvs)
		}},
		{"run_above_range", func(cvs *[256]byte, _, _ *sponge) bool {
			return tryProcessLeavesRunArch(input[:8*ChunkSize], 8, cvs)
		}},
		{"s0_below_range", func(cvs *[256]byte, final, _ *sponge) bool {
			return tryProcessS0LeavesArch(input[:ChunkSize], 1, final, cvs)
		}},
		{"s0_tail_below_range", func(cvs *[256]byte, final, partial *sponge) bool {
			return tryProcessS0LeavesTailArch(input[:ChunkSize+rate], 1, 1, final, partial, cvs)
		}},
		{"leaf_tail_below_range", func(cvs *[256]byte, _, partial *sponge) bool {
			return tryProcessLeavesTailArch(input[:rate], 0, 1, cvs, partial)
		}},
		{"leaf_tail_above_range", func(cvs *[256]byte, _, partial *sponge) bool {
			return tryProcessLeavesTailArch(input, 8, 1, cvs, partial)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assertTryKernelFailure(t, "invalid-shape", tc.try)
		})
	}
}

func processLeavesGeneric(input []byte, cvs *[256]byte) {
	for inst := range 8 {
		var s sponge
		off := inst * ChunkSize
		s.absorbAll(input[off:off+ChunkSize], leafDS)
		// Extract CV = first 4 lanes (32 bytes).
		s.squeeze(cvs[inst*32 : inst*32+32])
	}
}

func TestProcessLeaves(t *testing.T) {
	const blockSize = 8192

	// Build deterministic input: 8 × 8192 bytes.
	input := make([]byte, 8*blockSize)
	for i := range input {
		input[i] = byte(i*7 + i>>8)
	}

	// Compute expected CVs via generic path.
	var want [256]byte
	processLeavesGeneric(input, &want)

	// Compute CVs via the arch kernel.
	var got [256]byte
	if !tryProcessLeavesX8Arch(input, &got) {
		t.Skip("no x8 kernel on this platform")
	}

	if got != want {
		for inst := range 8 {
			wantCV := want[inst*32 : inst*32+32]
			gotCV := got[inst*32 : inst*32+32]
			for lane := range 4 {
				w := binary.LittleEndian.Uint64(wantCV[lane*8:])
				g := binary.LittleEndian.Uint64(gotCV[lane*8:])
				if w != g {
					t.Errorf("instance %d, lane %d: got %016x, want %016x", inst, lane, g, w)
				}
			}
		}
	}
}

func BenchmarkProcessLeaves(b *testing.B) {
	const blockSize = 8192
	input := make([]byte, 8*blockSize)
	for i := range input {
		input[i] = byte(i)
	}
	var cvs [256]byte
	if !tryProcessLeavesX8Arch(input, &cvs) {
		b.Skip("no x8 kernel on this platform")
	}
	b.SetBytes(8 * blockSize)
	for b.Loop() {
		tryProcessLeavesX8Arch(input, &cvs)
	}
}

// checkLeafCVs verifies each 32-byte chain value in cvs against the x1 leaf
// path for the corresponding chunk of input. prefix labels failures in tests
// that loop over shapes.
func checkLeafCVs(t *testing.T, prefix string, input, cvs []byte, n int) {
	t.Helper()
	for inst := range n {
		var s sponge
		leafStateX1(input[inst*ChunkSize:(inst+1)*ChunkSize], &s)
		var want [32]byte
		s.squeeze(want[:])
		if !bytes.Equal(cvs[inst*32:inst*32+32], want[:]) {
			t.Errorf("%sinstance %d: CV got %x, want %x", prefix, inst, cvs[inst*32:inst*32+32], want[:])
		}
	}
}

// TestProcessLeavesPair checks the 2-wide pair kernel against the x1 leaf path.
func TestProcessLeavesPair(t *testing.T) {
	input := make([]byte, 2*ChunkSize)
	for i := range input {
		input[i] = byte(i*31 + i>>7)
	}

	var got [256]byte
	if !tryProcessLeavesPairArch(input, &got) {
		t.Skip("no pair kernel on this platform")
	}

	checkLeafCVs(t, "", input, got[:], 2)
}

// TestProcessLeavesBatch5 checks the hybrid scalar/NEON 5-leaf kernel against
// the x1 leaf path. The scalar lane (chunk 4) and both NEON pairs must all
// produce correct CVs.
func TestProcessLeavesBatch5(t *testing.T) {
	input := make([]byte, 5*ChunkSize)
	for i := range input {
		input[i] = byte(i*37 + i>>7)
	}

	var got [256]byte
	if !tryProcessLeavesBatch5Arch(input, &got) {
		t.Skip("no batch5 kernel on this platform")
	}

	checkLeafCVs(t, "", input, got[:], 5)
}

func BenchmarkProcessLeavesBatch5(b *testing.B) {
	input := make([]byte, 5*ChunkSize)
	for i := range input {
		input[i] = byte(i)
	}
	var cvs [256]byte
	if !tryProcessLeavesBatch5Arch(input, &cvs) {
		b.Skip("no batch5 kernel on this platform")
	}
	b.SetBytes(5 * ChunkSize)
	for b.Loop() {
		tryProcessLeavesBatch5Arch(input, &cvs)
	}
}

func TestProcessLeavesTriple(t *testing.T) {
	input := make([]byte, 3*ChunkSize)
	for i := range input {
		input[i] = byte(i*41 + i>>7)
	}

	var got [256]byte
	if !tryProcessLeavesTripleArch(input, &got) {
		t.Skip("no x3 kernel on this platform")
	}
	checkLeafCVs(t, "", input, got[:], 3)
}

func BenchmarkProcessLeavesTriple(b *testing.B) {
	input := make([]byte, 3*ChunkSize)
	for i := range input {
		input[i] = byte(i)
	}
	var cvs [256]byte
	if !tryProcessLeavesTripleArch(input, &cvs) {
		b.Skip("no x3 kernel on this platform")
	}
	b.SetBytes(3 * ChunkSize)
	for b.Loop() {
		tryProcessLeavesTripleArch(input, &cvs)
	}
}

// TestProcessLeavesTail checks the trailing-leaves+partial kernel against the
// x1 leaf path across lane counts and head lengths spanning rate-block
// boundaries. arm64 hosts exactly n == 1; AVX-512 hosts n in 1..7.

func TestProcessS0Leaves(t *testing.T) {
	ran := false
	for n := 2; n <= availableLanes; n++ {
		input := make([]byte, n*ChunkSize)
		for i := range input {
			input[i] = byte(i*13 + i>>6 + n)
		}

		var final sponge
		var cvs [256]byte
		if !tryProcessS0LeavesArch(input, n, &final, &cvs) {
			continue
		}
		ran = true

		var wantFinal sponge
		wantFinal.absorb(input[:ChunkSize])
		wantFinal.absorb(kt12Marker[:])
		if final != wantFinal {
			t.Errorf("n=%d: final-node state:\n got %x pos=%d\nwant %x pos=%d",
				n, final.a, final.pos, wantFinal.a, wantFinal.pos)
		}

		checkLeafCVs(t, fmt.Sprintf("n=%d: ", n), input[ChunkSize:], cvs[32:], n-1)
	}
	if !ran {
		t.Skip("no fused S0+leaves kernel on this platform")
	}
}

// TestProcessS0LeavesTail checks the fused S_0+leaves+partial kernel against
// the serial paths across chunk counts and tail lengths spanning rate-block
// boundaries: the final-node state must match absorbing S_0 || kt12 marker,
// each complete leaf's CV must match the x1 path, and continuing the exported
// partial state must match a direct sponge over the full tail. The body is a
// helper so the amd64 tests can rerun it with the AVX2 kernels forced.

func TestProcessLeavesRun(t *testing.T) {
	for n := 2; n <= 7; n++ {
		input := make([]byte, n*ChunkSize)
		for i := range input {
			input[i] = byte(i*53 + i>>9)
		}

		var got [256]byte
		if !tryProcessLeavesRunArch(input, n, &got) {
			t.Skipf("no run kernel on this platform")
		}

		checkLeafCVs(t, fmt.Sprintf("n=%d: ", n), input, got[:], n)
	}
}

// BenchmarkLeafBatchRemainder measures processLeafBatch for the leftover-leaf
// counts that hit the remainder path during finalization.
func BenchmarkLeafBatchRemainder(b *testing.B) {
	for _, n := range []int{3, 5, 6, 7, 8, 13} {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			input := make([]byte, n*ChunkSize)
			for i := range input {
				input[i] = byte(i)
			}
			h := New(nil)
			h.state = stateTree
			b.SetBytes(int64(n * ChunkSize))
			for b.Loop() {
				h.final.reset()
				h.processLeafBatch(input, n)
			}
		})
	}
}

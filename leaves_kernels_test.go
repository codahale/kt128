package kt128

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

func processLeavesGeneric(input []byte, cvs *[256]byte) {
	for inst := range 8 {
		var s sponge
		off := inst * BlockSize
		s.absorbAll(input[off:off+BlockSize], leafDS)
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
	if !processLeavesArch(input, &got) {
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
	if !processLeavesArch(input, &cvs) {
		b.Skip("no x8 kernel on this platform")
	}
	b.SetBytes(8 * blockSize)
	for b.Loop() {
		processLeavesArch(input, &cvs)
	}
}

// checkLeafCVs verifies each 32-byte chain value in cvs against the x1 leaf
// path for the corresponding chunk of input. prefix labels failures in tests
// that loop over shapes.
func checkLeafCVs(t *testing.T, prefix string, input, cvs []byte, n int) {
	t.Helper()
	for inst := range n {
		var s sponge
		leafStateX1(input[inst*BlockSize:(inst+1)*BlockSize], &s)
		var want [32]byte
		s.squeeze(want[:])
		if !bytes.Equal(cvs[inst*32:inst*32+32], want[:]) {
			t.Errorf("%sinstance %d: CV got %x, want %x", prefix, inst, cvs[inst*32:inst*32+32], want[:])
		}
	}
}

// TestProcessLeavesPair checks the 2-wide pair kernel against the x1 leaf path.
func TestProcessLeavesPair(t *testing.T) {
	input := make([]byte, 2*BlockSize)
	for i := range input {
		input[i] = byte(i*31 + i>>7)
	}

	var got [256]byte
	if !processLeavesPairArch(input, &got) {
		t.Skip("no pair kernel on this platform")
	}

	checkLeafCVs(t, "", input, got[:], 2)
}

// TestProcessLeavesBatch5 checks the hybrid scalar/NEON 5-leaf kernel against
// the x1 leaf path. The scalar lane (chunk 4) and both NEON pairs must all
// produce correct CVs.
func TestProcessLeavesBatch5(t *testing.T) {
	input := make([]byte, 5*BlockSize)
	for i := range input {
		input[i] = byte(i*37 + i>>7)
	}

	var got [256]byte
	if !processLeavesBatch5Arch(input, &got) {
		t.Skip("no batch5 kernel on this platform")
	}

	checkLeafCVs(t, "", input, got[:], 5)
}

func BenchmarkProcessLeavesBatch5(b *testing.B) {
	input := make([]byte, 5*BlockSize)
	for i := range input {
		input[i] = byte(i)
	}
	var cvs [256]byte
	if !processLeavesBatch5Arch(input, &cvs) {
		b.Skip("no batch5 kernel on this platform")
	}
	b.SetBytes(5 * BlockSize)
	for b.Loop() {
		processLeavesBatch5Arch(input, &cvs)
	}
}

func TestProcessLeavesTriple(t *testing.T) {
	input := make([]byte, 3*BlockSize)
	for i := range input {
		input[i] = byte(i*41 + i>>7)
	}

	var got [256]byte
	if !processLeavesTripleArch(input, &got) {
		t.Skip("no x3 kernel on this platform")
	}
	checkLeafCVs(t, "", input, got[:], 3)
}

func BenchmarkProcessLeavesTriple(b *testing.B) {
	input := make([]byte, 3*BlockSize)
	for i := range input {
		input[i] = byte(i)
	}
	var cvs [256]byte
	if !processLeavesTripleArch(input, &cvs) {
		b.Skip("no x3 kernel on this platform")
	}
	b.SetBytes(3 * BlockSize)
	for b.Loop() {
		processLeavesTripleArch(input, &cvs)
	}
}

// TestProcessLeavesTail checks the trailing-leaves+partial kernel against the
// x1 leaf path across lane counts and head lengths spanning rate-block
// boundaries. arm64 hosts exactly n == 1; AVX-512 hosts n in 1..7.

func TestProcessS0Leaves(t *testing.T) {
	ran := false
	for n := 2; n <= availableLanes; n++ {
		input := make([]byte, n*BlockSize)
		for i := range input {
			input[i] = byte(i*13 + i>>6 + n)
		}

		var final sponge
		var cvs [256]byte
		if !processS0LeavesArch(input, n, &final, &cvs) {
			continue
		}
		ran = true

		var wantFinal sponge
		wantFinal.absorb(input[:BlockSize])
		wantFinal.absorb(kt12Marker[:])
		if final != wantFinal {
			t.Errorf("n=%d: final-node state:\n got %x pos=%d\nwant %x pos=%d",
				n, final.a, final.pos, wantFinal.a, wantFinal.pos)
		}

		checkLeafCVs(t, fmt.Sprintf("n=%d: ", n), input[BlockSize:], cvs[32:], n-1)
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
		input := make([]byte, n*BlockSize)
		for i := range input {
			input[i] = byte(i*53 + i>>9)
		}

		var got [256]byte
		if !processLeavesRunArch(input, n, &got) {
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
			input := make([]byte, n*BlockSize)
			for i := range input {
				input[i] = byte(i)
			}
			h := New(nil)
			h.state = stateTree
			b.SetBytes(int64(n * BlockSize))
			for b.Loop() {
				h.final.reset()
				h.leafCount = 0
				h.processLeafBatch(input, n)
			}
		})
	}
}

package kt128

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

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

	// Compute CVs via arch-dispatched path.
	var got [256]byte
	processLeaves(input, &got)

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
	b.SetBytes(8 * blockSize)
	for b.Loop() {
		processLeaves(input, &cvs)
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

	for inst := range 2 {
		var s sponge
		leafStateX1(input[inst*BlockSize:(inst+1)*BlockSize], &s)
		var want [256]byte
		s.squeeze(want[:32])
		if !bytes.Equal(got[inst*32:inst*32+32], want[:32]) {
			t.Errorf("instance %d: got %x, want %x", inst, got[inst*32:inst*32+32], want[:32])
		}
	}
}

// TestProcessS0Leaves checks the fused S_0+leaves kernel against the x1 paths
// for every chunk count: the final-node state must match absorbing
// S_0 || kt12 marker into a fresh sponge, and each chain value must match the
// x1 leaf path.
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

		for leaf := 1; leaf < n; leaf++ {
			var s sponge
			leafStateX1(input[leaf*BlockSize:(leaf+1)*BlockSize], &s)
			var want [32]byte
			s.squeeze(want[:])
			if !bytes.Equal(cvs[leaf*32:leaf*32+32], want[:]) {
				t.Errorf("n=%d leaf %d: CV got %x, want %x", n, leaf, cvs[leaf*32:leaf*32+32], want[:])
			}
		}
	}
	if !ran {
		t.Skip("no fused S0+leaves kernel on this platform")
	}
}

// TestProcessLeavesRun checks the direct-read run kernel against the x1 leaf
// path for every remainder size it handles.
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

		for inst := range n {
			var s sponge
			leafStateX1(input[inst*BlockSize:(inst+1)*BlockSize], &s)
			var want [256]byte
			s.squeeze(want[:32])
			if !bytes.Equal(got[inst*32:inst*32+32], want[:32]) {
				t.Errorf("n=%d instance %d: got %x, want %x", n, inst, got[inst*32:inst*32+32], want[:32])
			}
		}
	}
}

// BenchmarkLeafBatchRemainder measures processLeafBatch for the leftover-leaf
// counts that hit the remainder path during finalization.
func BenchmarkLeafBatchRemainder(b *testing.B) {
	for _, n := range []int{5, 6, 7} {
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

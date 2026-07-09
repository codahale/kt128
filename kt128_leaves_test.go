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

	for inst := range 5 {
		var s sponge
		leafStateX1(input[inst*BlockSize:(inst+1)*BlockSize], &s)
		var want [32]byte
		s.squeeze(want[:])
		if !bytes.Equal(got[inst*32:inst*32+32], want[:]) {
			t.Errorf("instance %d: got %x, want %x", inst, got[inst*32:inst*32+32], want[:])
		}
	}
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

// TestProcessLeavesTail checks the trailing-leaves+partial kernel against the
// x1 leaf path across lane counts and head lengths spanning rate-block
// boundaries. arm64 hosts exactly n == 1; AVX-512 hosts n in 1..7.
func TestProcessLeavesTail(t *testing.T) {
	suffix := []byte{0xA5, 0x5A, 0x03}
	ran := false
	for n := 1; n <= 7; n++ {
		for _, headLen := range []int{0, 1, 7, 8, 167, 168, 169, 4096, 8063, 8064, 8188} {
			input := make([]byte, n*BlockSize+headLen)
			for i := range input {
				input[i] = byte(i*17 + i>>5 + headLen + n)
			}
			head := input[n*BlockSize:]

			nShared := headLen / rate
			var cvs [256]byte
			var s sponge
			if !processLeavesTailArch(input, n, nShared, &cvs, &s) {
				continue
			}
			ran = true

			// Each complete leaf's CV must match the x1 path.
			for inst := range n {
				var ref sponge
				leafStateX1(input[inst*BlockSize:(inst+1)*BlockSize], &ref)
				var want [32]byte
				ref.squeeze(want[:])
				if !bytes.Equal(cvs[inst*32:inst*32+32], want[:]) {
					t.Errorf("n=%d headLen=%d: CV %d got %x, want %x", n, headLen, inst, cvs[inst*32:inst*32+32], want[:])
				}
			}

			// Finishing the exported partial state must match a direct sponge
			// over head || suffix.
			s.absorb(head[nShared*rate:])
			s.absorb(suffix)
			s.padPermute(leafDS)
			var direct sponge
			direct.absorb(head)
			direct.absorb(suffix)
			direct.padPermute(leafDS)
			if s != direct {
				t.Errorf("n=%d headLen=%d: partial leaf state diverges from direct absorption", n, headLen)
			}
		}
	}
	if !ran {
		t.Skip("no trailing-leaves+partial kernel on this platform")
	}
}

// TestPartialLeafFusionSizes cross-checks Write/Read against the RFC 9861
// reference for shapes whose finalization strands one or three complete leaves
// plus a ragged partial leaf — the partial-leaf fusion cases — and their
// boundary neighbors.
func TestPartialLeafFusionSizes(t *testing.T) {
	for _, custom := range [][]byte{nil, []byte("domain")} {
		for _, size := range []int{
			3*BlockSize + 1, 3*BlockSize + BlockSize/2, 4*BlockSize - 1, 4 * BlockSize,
			4*BlockSize + BlockSize/2, 5*BlockSize + BlockSize/2,
			7*BlockSize + BlockSize/2, 8 * BlockSize, 8*BlockSize + 1,
			// Finalization remainders of 1..7 complete leaves plus a partial
			// (amd64 tail-lane fusion; batched or serial elsewhere).
			10*BlockSize + BlockSize/2, 12*BlockSize + BlockSize/2,
			14*BlockSize + BlockSize/2, 16*BlockSize + BlockSize/2,
			17*BlockSize + 200, 19*BlockSize + rate - 1,
		} {
			msg := ptn(size)
			for _, chunk := range []int{size, BlockSize, BlockSize - 1} {
				h := New(custom)
				for off := 0; off < len(msg); off += chunk {
					_, _ = h.Write(msg[off:min(off+chunk, len(msg))])
				}
				got := make([]byte, 32)
				_, _ = h.Read(got)
				if want := referenceKT128(msg, custom, 32); !bytes.Equal(got, want) {
					t.Errorf("size=%d chunk=%d custom=%q: got %x, want %x", size, chunk, custom, got, want)
				}
			}
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

package kt128

import (
	"bytes"
	"fmt"
	"testing"
)

func TestProcessLeavesTail(t *testing.T) {
	suffix := []byte{0xA5, 0x5A, 0x03}
	ran := false
	for n := 1; n <= 7; n++ {
		for _, headLen := range []int{0, 1, 7, 8, 167, 168, 169, 4096, 8063, 8064, 8188} {
			input := make([]byte, n*ChunkSize+headLen)
			for i := range input {
				input[i] = byte(i*17 + i>>5 + headLen + n)
			}
			head := input[n*ChunkSize:]

			nShared := headLen / rate
			var cvs [256]byte
			var s sponge
			if !tryProcessLeavesTailArch(input, n, nShared, &cvs, &s) {
				continue
			}
			ran = true

			// Each complete leaf's CV must match the x1 path.
			checkLeafCVs(t, fmt.Sprintf("n=%d headLen=%d: ", n, headLen), input, cvs[:], n)

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
			3*ChunkSize + 1, 3*ChunkSize + ChunkSize/2, 4*ChunkSize - 1, 4 * ChunkSize,
			3*ChunkSize + 32*rate - 1, 3*ChunkSize + 32*rate,
			5*ChunkSize + 32*rate - 1, 5*ChunkSize + 32*rate,
			4*ChunkSize + ChunkSize/2, 5*ChunkSize + ChunkSize/2,
			7*ChunkSize + ChunkSize/2, 8 * ChunkSize, 8*ChunkSize + 1,
			// Finalization remainders of 1..7 complete leaves plus a partial
			// (amd64 tail-lane fusion; batched or serial elsewhere).
			10*ChunkSize + ChunkSize/2, 12*ChunkSize + ChunkSize/2,
			14*ChunkSize + ChunkSize/2, 16*ChunkSize + ChunkSize/2,
			17*ChunkSize + 200, 19*ChunkSize + rate - 1,
			// Chunk counts of 1 mod 8, on both sides of the whole-rate-block
			// tail threshold that decides amd64 S_0 fusion.
			9*ChunkSize + rate, 9*ChunkSize + rate - 1, 17*ChunkSize + 100,
			// Ragged one-shots whose partial tail rides the fused S_0 pass
			// (AVX-512), including both sides of the two-chunk threshold.
			2*ChunkSize + 26*rate, 2*ChunkSize + 27*rate, 2*ChunkSize + 8191,
			3*ChunkSize + rate, 6*ChunkSize + 4096, 7*ChunkSize + 8191,
		} {
			msg := ptn(size)
			for _, chunk := range []int{size, ChunkSize, ChunkSize - 1} {
				h := New(NewCustomizationString(custom))
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

// TestWritePartialLeafContinuation cross-checks writes that continue a fused
// first write's partial leaf against the RFC 9861 reference: the
// first write's shape triggers S_0+tail fusion (on platforms that have it),
// and the continuation must be equivalent whether it leaves the leaf
// incomplete, completes it exactly (or one byte around that boundary), or
// runs past it into more leaves. A clone taken mid-leaf must continue
// identically. The body is a helper so the amd64 tests can rerun it with the
// AVX2 kernels forced.
func TestWritePartialLeafContinuation(t *testing.T) { testWritePartialLeafContinuation(t) }

func testWritePartialLeafContinuation(t *testing.T) {
	for _, first := range []int{2*ChunkSize + 4096, 3*ChunkSize + 4096, 3*ChunkSize + 4200, 7*ChunkSize + 8191} {
		room := ChunkSize - first%ChunkSize
		for _, cont := range []int{0, 1, 100, 4096, room - 1, room, room + 1, room + ChunkSize, room + 9*ChunkSize + 17} {
			msg := ptn(first + cont)
			want := referenceKT128(msg, nil, 32)

			h := New(nil)
			_, _ = h.Write(msg[:first])
			clone := h.Clone()
			_, _ = h.Write(msg[first:])
			got := make([]byte, 32)
			_, _ = h.Read(got)
			if !bytes.Equal(got, want) {
				t.Errorf("first=%d cont=%d: split writes diverge: got %x, want %x", first, cont, got, want)
			}

			_, _ = clone.Write(msg[first:])
			cloned := make([]byte, 32)
			_, _ = clone.Read(cloned)
			if !bytes.Equal(cloned, want) {
				t.Errorf("first=%d cont=%d: cloned continuation diverges: got %x, want %x", first, cont, cloned, want)
			}
		}
	}
}

// TestProcessS0Leaves checks the fused S_0+leaves kernel against the x1 paths
// for every chunk count: the final-node state must match absorbing
// S_0 || kt12 marker into a fresh sponge, and each chain value must match the
// x1 leaf path.

func TestProcessS0LeavesTail(t *testing.T) { testProcessS0LeavesTail(t) }

func testProcessS0LeavesTail(t *testing.T) {
	suffix := []byte{0xC3, 0x3C, 0x0F}
	ran := false
	for n := 2; n <= 7; n++ {
		for _, tailLen := range []int{0, 1, 167, 168, 169, 4096, 8063, 8064, 8188} {
			input := make([]byte, n*ChunkSize+tailLen)
			for i := range input {
				input[i] = byte(i*19 + i>>7 + tailLen + n)
			}
			tail := input[n*ChunkSize:]
			nShared := tailLen / rate

			var final, pending sponge
			var cvs [256]byte
			if !tryProcessS0LeavesTailArch(input, n, nShared, &final, &pending, &cvs) {
				continue
			}
			ran = true

			var wantFinal sponge
			wantFinal.absorb(input[:ChunkSize])
			wantFinal.absorb(kt12Marker[:])
			if final != wantFinal {
				t.Errorf("n=%d tailLen=%d: final-node state:\n got %x pos=%d\nwant %x pos=%d",
					n, tailLen, final.a, final.pos, wantFinal.a, wantFinal.pos)
			}

			checkLeafCVs(t, fmt.Sprintf("n=%d tailLen=%d: ", n, tailLen), input[ChunkSize:], cvs[32:], n-1)

			// Finishing the exported partial state must match a direct sponge
			// over tail || suffix.
			pending.absorb(tail[nShared*rate:])
			pending.absorb(suffix)
			pending.padPermute(leafDS)
			var direct sponge
			direct.absorb(tail)
			direct.absorb(suffix)
			direct.padPermute(leafDS)
			if pending != direct {
				t.Errorf("n=%d tailLen=%d: partial leaf state diverges from direct absorption", n, tailLen)
			}
		}
	}
	if !ran {
		t.Skip("no fused S0+leaves+partial kernel on this platform")
	}
}

// TestProcessLeavesRun checks the direct-read run kernel against the x1 leaf
// path for every remainder size it handles.

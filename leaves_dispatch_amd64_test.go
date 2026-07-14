//go:build amd64 && !purego

package kt128

import (
	"fmt"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

func TestWriteForceAVX2DirectFlush(t *testing.T) {
	saved := cpuid.HasAVX512
	defer func() { cpuid.HasAVX512 = saved }()
	cpuid.HasAVX512 = false

	t.Run("quad tail flushes in place", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(8 * BlockSize)) // S_0+3 leaves fused, 4 leaves in place

		if h.leafCount != 7 {
			t.Fatalf("leaf count = %d, want 7", h.leafCount)
		}
		if cap(h.buf) != 0 {
			t.Fatalf("buffer capacity = %d, want 0", cap(h.buf))
		}
	})

	t.Run("sub-batch flush with buffered tail", func(t *testing.T) {
		h := New(nil)
		// S_0+3 leaves fused, 4 leaves in place, 3 chunks + 37 bytes buffered.
		_, _ = h.Write(ptn(11*BlockSize + 37))

		if h.leafCount != 7 {
			t.Fatalf("leaf count = %d, want 7", h.leafCount)
		}
		if len(h.buf) != 3*BlockSize+37 {
			t.Fatalf("buffered bytes = %d, want %d", len(h.buf), 3*BlockSize+37)
		}
	})
}

// TestWriteS0TailFusion pins the AVX-512 S_0+tail fused scheduling: a ragged
// one-shot first write of 2..7 chunks rides the partial's whole rate-blocks
// in an idle lane of the fused pass, leaving a pending leaf and only the
// ragged remnant buffered. Output correctness for these shapes is covered by
// TestPartialLeafFusionSizes and TestWritePendingContinuation.
func TestWriteS0TailFusion(t *testing.T) {
	if !cpuid.HasAVX512 {
		t.Skip("no AVX-512")
	}

	t.Run("ragged one-shot leaves a pending leaf", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(3*BlockSize + 4096)) // S_0+2 leaves fused, 24 tail blocks ride

		if h.pendingLen != 24*rate {
			t.Fatalf("pendingLen = %d, want %d", h.pendingLen, 24*rate)
		}
		if want := 4096 - 24*rate; len(h.buf) != want {
			t.Fatalf("buffered bytes = %d, want %d", len(h.buf), want)
		}
		if h.leafCount != 2 {
			t.Fatalf("leaf count = %d, want 2", h.leafCount)
		}
	})

	t.Run("two chunks below the pair threshold stay serial", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(2*BlockSize + (s0TailPairMin-1)*rate))

		if h.pendingLen != 0 {
			t.Fatalf("pendingLen = %d, want 0", h.pendingLen)
		}
	})

	t.Run("two chunks at the pair threshold ride the quad", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(2*BlockSize + s0TailPairMin*rate))

		if h.pendingLen != s0TailPairMin*rate {
			t.Fatalf("pendingLen = %d, want %d", h.pendingLen, s0TailPairMin*rate)
		}
	})

	t.Run("eight chunks have no free lane", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(8*BlockSize + 4096))

		if h.pendingLen != 0 {
			t.Fatalf("pendingLen = %d, want 0", h.pendingLen)
		}
	})
}

// TestS0TailFusionForceAVX2 reruns the S_0+tail kernel differential and
// pending-continuation tests with the AVX2 quad kernels forced, so both
// kernel families are exercised on an AVX-512 host.
func TestS0TailFusionForceAVX2(t *testing.T) {
	if !cpuid.HasAVX512 {
		t.Skip("AVX2 path already exercised natively")
	}
	defer func() { cpuid.HasAVX512 = true }()
	cpuid.HasAVX512 = false
	t.Run("kernel", testProcessS0LeavesTail)
	t.Run("continuation", testWritePendingContinuation)
}

// TestWriteS0TailFusionAVX2 pins the AVX2 S_0+tail fused scheduling: a
// ragged one-shot first write of two or three chunks rides the partial's
// whole rate-blocks in the quad's free lane unconditionally; four chunks
// fill the quad and leave no lane.
func TestWriteS0TailFusionAVX2(t *testing.T) {
	saved := cpuid.HasAVX512
	defer func() { cpuid.HasAVX512 = saved }()
	cpuid.HasAVX512 = false

	t.Run("two-chunk ragged one-shot leaves a pending leaf", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(2*BlockSize + 4096)) // S_0+1 leaf fused, 24 tail blocks ride

		if h.pendingLen != 24*rate {
			t.Fatalf("pendingLen = %d, want %d", h.pendingLen, 24*rate)
		}
		if want := 4096 - 24*rate; len(h.buf) != want {
			t.Fatalf("buffered bytes = %d, want %d", len(h.buf), want)
		}
		if h.leafCount != 1 {
			t.Fatalf("leaf count = %d, want 1", h.leafCount)
		}
	})

	t.Run("three-chunk ragged one-shot leaves a pending leaf", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(3*BlockSize + 300)) // one whole tail block rides

		if h.pendingLen != rate {
			t.Fatalf("pendingLen = %d, want %d", h.pendingLen, rate)
		}
		if want := 300 - rate; len(h.buf) != want {
			t.Fatalf("buffered bytes = %d, want %d", len(h.buf), want)
		}
	})

	t.Run("four chunks have no free lane", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(4*BlockSize + 4096))

		if h.pendingLen != 0 {
			t.Fatalf("pendingLen = %d, want 0", h.pendingLen)
		}
	})
}

// BenchmarkWriteForceAVX2 measures one-shot hashing with the AVX2 kernels forced
// (HasAVX512 disabled), so the AVX2 remainder path is exercised on this host.
func BenchmarkWriteForceAVX2(b *testing.B) {
	saved := cpuid.HasAVX512
	defer func() { cpuid.HasAVX512 = saved }()
	for _, size := range []int{32 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024} {
		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			msg := ptn(size)
			out := make([]byte, 32)
			b.SetBytes(int64(size))
			cpuid.HasAVX512 = false
			for b.Loop() {
				h := New(nil)
				_, _ = h.Write(msg)
				_, _ = h.Read(out)
			}
		})
	}
}

// TestAVX2MatchesAVX512 hashes a range of message/customization sizes (clustered
// around chunk and SIMD-batch boundaries, so every remainder path is exercised)
// with the AVX2 kernels forced and confirms the output matches the AVX-512
// kernels. The AVX-512 path is itself validated against the RFC vectors in
// TestRFCVectors. The large and customized cases below reproduce, as a direct
// AVX-512-vs-AVX2 comparison, the shapes that diverged under SDE -skx so the
// failure is localized to the AVX-512 kernels rather than only seen end-to-end.

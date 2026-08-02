//go:build amd64 && !purego

package kt128

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

func TestWriteForceGenericFallback(t *testing.T) {
	savedAVX512, savedAVX2 := cpuid.HasAVX512, cpuid.HasAVX2
	savedHasLeafX8 := hasLeafX8
	defer func() {
		cpuid.HasAVX512, cpuid.HasAVX2 = savedAVX512, savedAVX2
		hasLeafX8 = savedHasLeafX8
	}()
	cpuid.HasAVX512, cpuid.HasAVX2 = false, false
	hasLeafX8 = false

	for _, size := range []int{
		0, ChunkSize, 2 * ChunkSize, 9*ChunkSize + 137, 1024 * 1024,
	} {
		t.Run(fmt.Sprintf("%d", size), func(t *testing.T) {
			msg := ptn(size)
			h := New([]byte("generic-fallback"))
			for off := 0; off < len(msg); off += 3333 {
				_, _ = h.Write(msg[off:min(off+3333, len(msg))])
			}
			got := make([]byte, 64)
			_, _ = h.Read(got)
			want := referenceKT128(msg, []byte("generic-fallback"), len(got))
			if !bytes.Equal(got, want) {
				t.Fatalf("generic fallback output %x != reference %x", got, want)
			}
		})
	}
}

func TestWriteForceAVX2DirectFlush(t *testing.T) {
	if !cpuid.HasAVX2 {
		t.Skip("no AVX2")
	}
	saved := cpuid.HasAVX512
	defer func() { cpuid.HasAVX512 = saved }()
	cpuid.HasAVX512 = false

	t.Run("quad tail flushes in place", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(8 * ChunkSize)) // S_0+3 leaves fused, 4 leaves in place

		if h.leafLen != 0 {
			t.Fatalf("partial leaf bytes = %d, want 0", h.leafLen)
		}
	})

	t.Run("sub-batch flush with partial leaf", func(t *testing.T) {
		h := New(nil)
		// S_0+3 leaves fuse; the remaining complete leaves process in place.
		_, _ = h.Write(ptn(11*ChunkSize + 37))

		if h.leafLen != 37 {
			t.Fatalf("partial leaf bytes = %d, want 37", h.leafLen)
		}
	})
}

// TestWriteS0TailFusion pins the AVX-512 S_0+tail fused scheduling: a ragged
// one-shot first write of 2..7 chunks rides the partial's whole rate-blocks
// in an idle lane of the fused pass, leaving an incremental partial leaf.
// Output correctness for these shapes is covered by
// TestPartialLeafFusionSizes and TestWritePartialLeafContinuation.
func TestWriteS0TailFusion(t *testing.T) {
	if !cpuid.HasAVX512 {
		t.Skip("no AVX-512")
	}

	t.Run("ragged one-shot leaves a partial leaf", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(3*ChunkSize + 4096)) // S_0+2 leaves fused, 24 tail blocks ride

		if h.leafLen != 4096 {
			t.Fatalf("partial leaf bytes = %d, want 4096", h.leafLen)
		}
	})

	t.Run("two chunks below the pair threshold stay serial", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(2*ChunkSize + (s0TailPairMin-1)*rate))

		if want := (s0TailPairMin - 1) * rate; h.leafLen != want {
			t.Fatalf("partial leaf bytes = %d, want %d", h.leafLen, want)
		}
	})

	t.Run("two chunks at the pair threshold ride the quad", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(2*ChunkSize + s0TailPairMin*rate))

		if h.leafLen != s0TailPairMin*rate {
			t.Fatalf("partial leaf bytes = %d, want %d", h.leafLen, s0TailPairMin*rate)
		}
	})

	t.Run("eight chunks have no free lane", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(8*ChunkSize + 4096))

		if h.leafLen != 4096 {
			t.Fatalf("partial leaf bytes = %d, want 4096", h.leafLen)
		}
	})
}

// TestS0TailFusionForceAVX2 reruns the S_0+tail kernel differential and
// partial-leaf continuation tests with the AVX2 quad kernels forced, so both
// kernel families are exercised on an AVX-512 host.
func TestS0TailFusionForceAVX2(t *testing.T) {
	if !cpuid.HasAVX2 {
		t.Skip("no AVX2")
	}
	if !cpuid.HasAVX512 {
		t.Skip("AVX2 path already exercised natively")
	}
	defer func() { cpuid.HasAVX512 = true }()
	cpuid.HasAVX512 = false
	t.Run("kernel", testProcessS0LeavesTail)
	t.Run("continuation", testWritePartialLeafContinuation)
}

// TestWriteS0TailFusionAVX2 pins the AVX2 S_0+tail fused scheduling: a
// ragged one-shot first write of two or three chunks rides the partial's
// whole rate-blocks in the quad's free lane unconditionally; four chunks
// fill the quad and leave no lane.
func TestWriteS0TailFusionAVX2(t *testing.T) {
	if !cpuid.HasAVX2 {
		t.Skip("no AVX2")
	}
	saved := cpuid.HasAVX512
	defer func() { cpuid.HasAVX512 = saved }()
	cpuid.HasAVX512 = false

	t.Run("two-chunk ragged one-shot leaves a partial leaf", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(2*ChunkSize + 4096)) // S_0+1 leaf fused, 24 tail blocks ride

		if h.leafLen != 4096 {
			t.Fatalf("partial leaf bytes = %d, want 4096", h.leafLen)
		}
	})

	t.Run("three-chunk ragged one-shot leaves a partial leaf", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(3*ChunkSize + 300)) // one whole tail block rides

		if h.leafLen != 300 {
			t.Fatalf("partial leaf bytes = %d, want 300", h.leafLen)
		}
	})

	t.Run("four chunks have no free lane", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(4*ChunkSize + 4096))

		if h.leafLen != 4096 {
			t.Fatalf("partial leaf bytes = %d, want 4096", h.leafLen)
		}
	})
}

// BenchmarkWriteForceAVX2 measures one-shot hashing with the AVX2 kernels forced
// (HasAVX512 disabled), so the AVX2 remainder path is exercised on this host.
func BenchmarkWriteForceAVX2(b *testing.B) {
	if !cpuid.HasAVX2 {
		b.Skip("no AVX2")
	}
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

//go:build amd64 && !purego

package kt128

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// TestProcessLeavesRunAVX2 forces the AVX2 run path and checks every remainder
// size it handles against the x1 leaf path.
func TestProcessLeavesRunAVX2(t *testing.T) {
	if !cpuid.HasAVX512 {
		return // already exercised by TestProcessLeavesRun
	}
	defer func() { cpuid.HasAVX512 = true }()
	cpuid.HasAVX512 = false

	for n := 2; n <= 7; n++ {
		input := make([]byte, n*BlockSize)
		for i := range input {
			input[i] = byte(i*53 + i>>9)
		}
		var got [256]byte
		if !processLeavesRunArch(input, n, &got) {
			t.Fatalf("AVX2 run kernel reported unavailable")
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
func TestAVX2MatchesAVX512(t *testing.T) {
	if !cpuid.HasAVX512 {
		t.Skip("no AVX-512 available to compare against")
	}

	// compare hashes msg with customization custom both ways — first AVX-512
	// (HasAVX512 true), then AVX2 (forced off) — and reports any divergence.
	compare := func(t *testing.T, msg, custom []byte) {
		t.Helper()

		ref := New(custom)
		_, _ = ref.Write(msg)
		want := make([]byte, 64)
		_, _ = ref.Read(want)

		cpuid.HasAVX512 = false
		h := New(custom)
		_, _ = h.Write(msg)
		got := make([]byte, 64)
		_, _ = h.Read(got)
		cpuid.HasAVX512 = true

		if !bytes.Equal(got, want) {
			t.Errorf("AVX2 output %x != AVX-512 output %x", got, want)
		}
	}

	sizes := []int{
		0, 1, BlockSize, BlockSize + 1,
		9 * BlockSize, 10 * BlockSize, 11 * BlockSize, 12 * BlockSize,
		13 * BlockSize, 14 * BlockSize, 15 * BlockSize,
		17*BlockSize + 123, 23*BlockSize + 4567, 64 * 1024, 1024 * 1024,
		2 * 1024 * 1024, 8 * 1024 * 1024,
		24137569, // the RFC vector that diverged under SDE -skx
	}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%d", size), func(t *testing.T) {
			compare(t, ptn(size), nil)
		})
	}

	// Customized inputs. {BlockSize, 2*BlockSize+3} is the shape that diverged in
	// TestWritePartitionInvariance under SDE -skx (two customization-suffix leaves
	// drained by the run kernel).
	customs := []struct{ msg, custom int }{
		{BlockSize, 2*BlockSize + 3},
		{1, BlockSize + 64},
		{3 * BlockSize, 5*BlockSize + 7},
	}
	for _, tc := range customs {
		t.Run(fmt.Sprintf("%d_c%d", tc.msg, tc.custom), func(t *testing.T) {
			compare(t, ptn(tc.msg), ptn(tc.custom))
		})
	}
}

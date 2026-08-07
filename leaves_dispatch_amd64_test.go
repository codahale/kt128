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
	defer func() {
		cpuid.HasAVX512, cpuid.HasAVX2 = savedAVX512, savedAVX2
	}()
	cpuid.HasAVX512, cpuid.HasAVX2 = false, false
	testUnavailableKernelWrappers(t)

	for _, size := range []int{
		0, ChunkSize, 2 * ChunkSize, 9*ChunkSize + 137, 1024 * 1024,
	} {
		t.Run(fmt.Sprintf("%d", size), func(t *testing.T) {
			msg := ptn(size)
			h := NewXOF([]byte("generic-fallback"))
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

func TestS0TailScheduling(t *testing.T) {
	savedAVX512, savedAVX2 := cpuid.HasAVX512, cpuid.HasAVX2
	defer func() {
		cpuid.HasAVX512, cpuid.HasAVX2 = savedAVX512, savedAVX2
	}()

	for _, tc := range []struct {
		name               string
		avx512, avx2       bool
		chunks, tail       int
		wantChunks         int
		wantTailRateBlocks int
	}{
		{"AVX-512 remainder", true, true, 3, 4096, 3, 4096 / rate},
		{"AVX-512 pair below crossover", true, true, 2, (s0TailPairMin - 1) * rate, 2, 0},
		{"AVX-512 pair at crossover", true, true, 2, s0TailPairMin * rate, 2, s0TailPairMin},
		{"AVX-512 full batch", true, true, 8, 4096, 8, 0},
		{"AVX2 pair", false, true, 2, 4096, 2, 4096 / rate},
		{"AVX2 triple", false, true, 3, 300, 3, 300 / rate},
		{"AVX2 full quad", false, true, 4, 4096, 4, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cpuid.HasAVX512, cpuid.HasAVX2 = tc.avx512, tc.avx2
			if got := fuseS0Chunks(tc.chunks, tc.tail); got != tc.wantChunks {
				t.Errorf("fuseS0Chunks(%d, %d) = %d, want %d", tc.chunks, tc.tail, got, tc.wantChunks)
			}
			if got := fuseS0TailBlocks(tc.chunks, tc.tail); got != tc.wantTailRateBlocks {
				t.Errorf("fuseS0TailBlocks(%d, %d) = %d, want %d", tc.chunks, tc.tail, got, tc.wantTailRateBlocks)
			}
		})
	}
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

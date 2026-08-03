//go:build amd64 && !purego

package kt128

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// TestProcessLeavesRunAVX2 forces the AVX2 run path and checks every remainder

func TestProcessLeavesRunAVX2(t *testing.T) {
	if !cpuid.HasAVX2 {
		t.Skip("no AVX2")
	}
	if !cpuid.HasAVX512 {
		return // already exercised by TestProcessLeavesRun
	}
	defer func() { cpuid.HasAVX512 = true }()
	cpuid.HasAVX512 = false

	for n := 2; n <= 7; n++ {
		input := make([]byte, n*ChunkSize)
		for i := range input {
			input[i] = byte(i*53 + i>>9)
		}
		var got [256]byte
		if !tryProcessLeavesRunArch(input, n, &got) {
			t.Fatalf("AVX2 run kernel reported unavailable")
		}
		checkLeafCVs(t, fmt.Sprintf("n=%d: ", n), input, got[:], n)
	}
}

// TestProcessLeavesPairAVX512 checks the 2-wide XMM pair kernel against the x1
// leaf path.
func TestProcessLeavesPairAVX512(t *testing.T) {
	if !cpuid.HasAVX512 {
		t.Skip("no AVX-512")
	}
	input := make([]byte, 2*ChunkSize)
	for i := range input {
		input[i] = byte(i*29 + i>>6)
	}
	var got [256]byte
	processLeavesPairAVX512(&input[0], &got[0])
	checkLeafCVs(t, "", input, got[:], 2)
}

// TestWriteForceAVX2DirectFlush pins the AVX2 direct-flush shapes: with a
// four-chunk flush unit, a quad-sized tail left after S_0 fusion flushes
// straight from the caller's buffer. Output correctness for these shapes is
// covered by
// TestAVX2MatchesAVX512; this test asserts the scheduling itself.

func TestAVX2MatchesAVX512(t *testing.T) {
	if !cpuid.HasAVX512 {
		t.Skip("no AVX-512 available to compare against")
	}
	if !cpuid.HasAVX2 {
		t.Skip("no AVX2 available to compare against")
	}

	// compare hashes msg with customization custom both ways — first AVX-512
	// (HasAVX512 true), then AVX2 (forced off) — and reports any divergence.
	compare := func(t *testing.T, msg, custom []byte) {
		t.Helper()

		customization := bytes.Clone(custom)
		ref := New(customization)
		_, _ = ref.Write(msg)
		want := make([]byte, 64)
		_, _ = ref.Read(want)

		cpuid.HasAVX512 = false
		h := New(customization)
		_, _ = h.Write(msg)
		got := make([]byte, 64)
		_, _ = h.Read(got)
		cpuid.HasAVX512 = true

		if !bytes.Equal(got, want) {
			t.Errorf("AVX2 output %x != AVX-512 output %x", got, want)
		}
	}

	sizes := []int{
		0, 1, ChunkSize, ChunkSize + 1,
		9 * ChunkSize, 10 * ChunkSize, 11 * ChunkSize, 12 * ChunkSize,
		13 * ChunkSize, 14 * ChunkSize, 15 * ChunkSize,
		17*ChunkSize + 123, 23*ChunkSize + 4567, 64 * 1024, 1024 * 1024,
		2 * 1024 * 1024, 8 * 1024 * 1024,
		24137569, // the RFC vector that diverged under SDE -skx
		// AVX2 S0-quad and quad-tail fusion shapes: 2..4-chunk messages,
		// finalization remainders of 1..3 completes plus a partial, and both
		// sides of the mod-8-equals-5 stranded-leaf exception.
		2 * ChunkSize, 3 * ChunkSize, 4 * ChunkSize,
		4*ChunkSize + ChunkSize/2, 6*ChunkSize + ChunkSize/2,
		9*ChunkSize + ChunkSize/2, 10*ChunkSize + ChunkSize/2,
		5 * ChunkSize, 5*ChunkSize + rate, 13 * ChunkSize, 13*ChunkSize + rate - 1,
		// S_0+tail fused shapes where the two families schedule differently:
		// at two chunks AVX-512 rides only past the pair threshold while the
		// AVX2 quad always rides; at three chunks both ride.
		2*ChunkSize + 300, 2*ChunkSize + 8191, 3*ChunkSize + 4096,
	}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%d", size), func(t *testing.T) {
			compare(t, ptn(size), nil)
		})
	}

	// Customized inputs. {ChunkSize, 2*ChunkSize+3} is the shape that diverged in
	// TestWritePartitionInvariance under SDE -skx (two customization-suffix leaves
	// drained by the run kernel).
	customs := []struct{ msg, custom int }{
		{ChunkSize, 2*ChunkSize + 3},
		{1, ChunkSize + 64},
		{3 * ChunkSize, 5*ChunkSize + 7},
	}
	for _, tc := range customs {
		t.Run(fmt.Sprintf("%d_c%d", tc.msg, tc.custom), func(t *testing.T) {
			compare(t, ptn(tc.msg), ptn(tc.custom))
		})
	}
}

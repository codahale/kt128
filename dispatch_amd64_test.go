//go:build amd64 && !purego

package kt128

import (
	"os"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// TestExpectedISA asserts that the SIMD path selected at runtime matches the one
// the environment intends to exercise. CI sets KT128_EXPECT_AVX512 to "1" (the
// SDE Skylake-X job, AVX-512) or "0" (the SDE Haswell job and the
// kt128_disable_avx512 build, AVX2 only). Without this, an SDE misconfiguration,
// a CPUID-detection bug, or a build-tag regression that lets real detection leak
// into the disabled build would pass while silently running the wrong kernels —
// a green check that tested nothing. Unset or empty (local runs, the standard CI
// job), it skips.
func TestExpectedISA(t *testing.T) {
	want := os.Getenv("KT128_EXPECT_AVX512")
	if want == "" {
		t.Skip("KT128_EXPECT_AVX512 unset; skipping ISA dispatch assertion")
	}

	switch want {
	case "1":
		if !cpuid.HasAVX512 {
			t.Fatal("KT128_EXPECT_AVX512=1 but cpuid.HasAVX512 is false: the AVX-512 kernels were not exercised")
		}
	case "0":
		if cpuid.HasAVX512 {
			t.Fatal("KT128_EXPECT_AVX512=0 but cpuid.HasAVX512 is true: the AVX2 path was not exercised")
		}
	default:
		t.Fatalf("KT128_EXPECT_AVX512=%q: want \"0\" or \"1\"", want)
	}
}

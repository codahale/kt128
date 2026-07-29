//go:build amd64 && !purego

package kt128

import (
	"os"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// TestExpectedISA asserts that the SIMD path selected at runtime matches the one
// the environment intends to exercise. CI sets KT128_EXPECT_AVX512 and
// KT128_EXPECT_AVX2 for its emulated CPUs and AVX-512-disabled build. Without
// this, an SDE misconfiguration, a CPUID-detection bug, or a build-tag regression
// could pass while silently running the wrong kernels. Unset or empty (local
// runs and the standard CI job), each assertion is skipped.
func TestExpectedISA(t *testing.T) {
	assertFeature := func(name, want string, got bool) {
		t.Helper()
		if want == "" {
			return
		}
		switch want {
		case "1":
			if !got {
				t.Fatalf("%s=1 but detected feature is false", name)
			}
		case "0":
			if got {
				t.Fatalf("%s=0 but detected feature is true", name)
			}
		default:
			t.Fatalf("%s=%q: want \"0\" or \"1\"", name, want)
		}
	}

	wantAVX512 := os.Getenv("KT128_EXPECT_AVX512")
	wantAVX2 := os.Getenv("KT128_EXPECT_AVX2")
	if wantAVX512 == "" && wantAVX2 == "" {
		t.Skip("ISA expectations unset; skipping dispatch assertion")
	}
	assertFeature("KT128_EXPECT_AVX512", wantAVX512, cpuid.HasAVX512)
	assertFeature("KT128_EXPECT_AVX2", wantAVX2, cpuid.HasAVX2)
}

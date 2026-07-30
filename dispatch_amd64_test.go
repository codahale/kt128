//go:build amd64 && !purego

package kt128

import (
	"os"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// TestExpectedISA asserts that the assembly paths selected at runtime match the
// ones the environment intends to exercise. CI sets KT128_EXPECT_AVX512,
// KT128_EXPECT_AVX2, and KT128_EXPECT_BMI2 for its emulated CPUs and
// AVX-512-disabled build. Without this, an SDE misconfiguration, a
// CPUID-detection bug, or a build-tag regression could pass while silently
// running the wrong kernels. Unset or empty (local runs and the standard CI
// job), each assertion is skipped.
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
	wantBMI2 := os.Getenv("KT128_EXPECT_BMI2")
	if wantAVX512 == "" && wantAVX2 == "" && wantBMI2 == "" {
		t.Skip("ISA expectations unset; skipping dispatch assertion")
	}
	assertFeature("KT128_EXPECT_AVX512", wantAVX512, cpuid.HasAVX512)
	assertFeature("KT128_EXPECT_AVX2", wantAVX2, cpuid.HasAVX2)
	assertFeature("KT128_EXPECT_BMI2", wantBMI2, cpuid.HasBMI2)
}

func TestScalarAssemblyDispatchRequiresBMI2(t *testing.T) {
	savedAVX512 := cpuid.HasAVX512
	savedBMI2 := cpuid.HasBMI2
	defer func() {
		cpuid.HasAVX512 = savedAVX512
		cpuid.HasBMI2 = savedBMI2
	}()

	cpuid.HasAVX512 = false
	cpuid.HasBMI2 = false
	var s sponge
	if permute12x1Arch(&s) {
		t.Fatal("permute12x1Arch selected scalar assembly without BMI2")
	}
	if fastLoopAbsorb168x1Arch(&s, make([]byte, rate)) {
		t.Fatal("fastLoopAbsorb168x1Arch selected scalar assembly without BMI2")
	}
}

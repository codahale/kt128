//go:build arm64 && !purego

package kt128

import (
	"os"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

func TestExpectedSHA3(t *testing.T) {
	want := os.Getenv("KT128_EXPECT_SHA3")
	if want == "" {
		t.Skip("KT128_EXPECT_SHA3 unset; skipping ISA dispatch assertion")
	}
	switch want {
	case "1":
		if !cpuid.HasSHA3 {
			t.Fatal("KT128_EXPECT_SHA3=1 but cpuid.HasSHA3 is false")
		}
	case "0":
		if cpuid.HasSHA3 {
			t.Fatal("KT128_EXPECT_SHA3=0 but cpuid.HasSHA3 is true")
		}
	default:
		t.Fatalf("KT128_EXPECT_SHA3=%q: want \"0\" or \"1\"", want)
	}
}

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

func TestRecommendedWriteBufferSizeARM64(t *testing.T) {
	savedSHA3 := cpuid.HasSHA3
	defer func() { cpuid.HasSHA3 = savedSHA3 }()

	for _, tc := range []struct {
		name             string
		sha3             bool
		wantBufferChunks int
	}{
		{"SHA3", true, 5},
		{"scalar", false, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cpuid.HasSHA3 = tc.sha3
			if got, want := RecommendedWriteBufferSize(), tc.wantBufferChunks*ChunkSize; got != want {
				t.Fatalf("RecommendedWriteBufferSize() = %d, want %d", got, want)
			}
		})
	}
}

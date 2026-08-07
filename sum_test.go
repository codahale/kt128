package kt128

import (
	"bytes"
	"fmt"
	"hash"
	"testing"
)

func TestHashSum(t *testing.T) {
	tests := []struct {
		name    string
		message []byte
		custom  []byte
	}{
		{name: "empty"},
		{name: "single node", message: ptn(ChunkSize)},
		{name: "tree", message: ptn(ChunkSize + 1)},
		{name: "customized", message: ptn(ChunkSize - 4), custom: []byte("domain")},
	}

	for _, digestSize := range []int{1, 16, 32, rate + 3} {
		for _, tc := range tests {
			t.Run(fmt.Sprintf("%s/digest=%d", tc.name, digestSize), func(t *testing.T) {
				h := NewHash(tc.custom, digestSize)
				_, _ = h.Write(tc.message)

				prefix := []byte("prefix")
				got := h.Sum(bytes.Clone(prefix))
				want := append(bytes.Clone(prefix), referenceKT128(tc.message, tc.custom, digestSize)...)
				if !bytes.Equal(got, want) {
					t.Fatalf("Sum() = %x, want %x", got, want)
				}
				if h.Pos() != uint64(len(tc.message)) {
					t.Fatalf("Sum changed Pos() to %d, want %d", h.Pos(), len(tc.message))
				}

				// Sum must leave the absorption state unchanged and repeatable.
				if next := h.Sum(nil); !bytes.Equal(next, want[len(prefix):]) {
					t.Fatalf("second Sum() = %x, want %x", next, want[len(prefix):])
				}
				_, _ = h.Write([]byte("tail"))
				wantAfterWrite := referenceKT128(append(bytes.Clone(tc.message), "tail"...), tc.custom, digestSize)
				if next := h.Sum(nil); !bytes.Equal(next, wantAfterWrite) {
					t.Fatalf("Sum() after Write = %x, want %x", next, wantAfterWrite)
				}
			})
		}
	}
}

func TestHashSize(t *testing.T) {
	for _, size := range []int{1, 16, 32, 64, 257} {
		if got := NewHash(nil, size).Size(); got != size {
			t.Fatalf("NewHash(nil, %d).Size() = %d", size, got)
		}
	}
	var zero Hash
	if got := zero.Size(); got != defaultDigestSize {
		t.Fatalf("zero-value Size() = %d, want %d", got, defaultDigestSize)
	}
}

func TestHashZeroValue(t *testing.T) {
	var h Hash
	message := []byte("message")
	_, _ = h.Write(message)
	if got, want := h.Sum(nil), referenceKT128(message, nil, defaultDigestSize); !bytes.Equal(got, want) {
		t.Fatalf("zero-value digest = %x, want %x", got, want)
	}
}

func TestHashResetPreservesConfiguration(t *testing.T) {
	custom := []byte("domain")
	h := NewHash(custom, 57)
	_, _ = h.Write([]byte("discarded"))
	h.Reset()
	message := []byte("message")
	_, _ = h.Write(message)
	if h.Size() != 57 {
		t.Fatalf("Size() after Reset = %d, want 57", h.Size())
	}
	if got, want := h.Sum(nil), referenceKT128(message, custom, 57); !bytes.Equal(got, want) {
		t.Fatalf("digest after Reset = %x, want %x", got, want)
	}
}

func TestNewHashPanicsWithNonPositiveDigestSize(t *testing.T) {
	for _, size := range []int{-1, 0} {
		mustPanic(t, "kt128: non-positive digest size", func() {
			NewHash(nil, size)
		})
	}
}

func TestHashAndXOFInterfacesAreDistinct(t *testing.T) {
	if _, ok := any(NewHash(nil, 32)).(hash.XOF); ok {
		t.Fatal("Hash implements hash.XOF")
	}
	if _, ok := any(NewXOF(nil)).(hash.Hash); ok {
		t.Fatal("XOF implements hash.Hash")
	}
}

func TestSum(t *testing.T) {
	tests := []struct {
		messageLen int
		custom     []byte
		outputLen  int
	}{
		{messageLen: 0, outputLen: 0},
		{messageLen: 1, outputLen: 1},
		{messageLen: ChunkSize - 1, outputLen: 16},
		{messageLen: ChunkSize, outputLen: 32},
		{messageLen: ChunkSize + 1, outputLen: 64},
		{messageLen: 9 * ChunkSize, outputLen: 257},
		{messageLen: ChunkSize - 4, custom: []byte("domain"), outputLen: 32},
		{messageLen: ChunkSize, custom: ptn(41), outputLen: 64},
	}

	for _, tc := range tests {
		name := fmt.Sprintf("message=%d/custom=%d/output=%d", tc.messageLen, len(tc.custom), tc.outputLen)
		t.Run(name, func(t *testing.T) {
			message := ptn(tc.messageLen)
			want := referenceKT128(message, tc.custom, tc.outputLen)
			got := Sum(message, tc.custom, tc.outputLen)
			if !bytes.Equal(got, want) {
				t.Fatalf("Sum() = %x, want %x", got, want)
			}
		})
	}
}

func TestSumDoesNotModifyInputs(t *testing.T) {
	message := ptn(ChunkSize + 1)
	customization := ptn(41)
	wantMessage := bytes.Clone(message)
	wantCustomization := bytes.Clone(customization)

	_ = Sum(message, customization, 32)

	if !bytes.Equal(message, wantMessage) {
		t.Fatal("Sum modified its message input")
	}
	if !bytes.Equal(customization, wantCustomization) {
		t.Fatal("Sum modified its customization input")
	}
}

func TestSumPanicsWithNegativeOutputLength(t *testing.T) {
	mustPanic(t, "kt128: negative output length", func() {
		Sum(nil, nil, -1)
	})
}

func TestSumRFCVector(t *testing.T) {
	want := mustHex("1AC2D450FC3B4205D19DA7BFCA1B37513C0803577AC7167F06FE2CE1F0EF39E5")
	got := Sum(nil, nil, 32)
	if !bytes.Equal(got, want) {
		t.Fatalf("Sum(nil, nil, 32) = %x, want %x", got, want)
	}
}

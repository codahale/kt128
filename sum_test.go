package kt128

import (
	"bytes"
	"fmt"
	"testing"
)

func TestHasherSum(t *testing.T) {
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

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := New(tc.custom)
			_, _ = h.Write(tc.message)

			prefix := []byte("prefix")
			got := h.Sum(bytes.Clone(prefix))
			want := append(bytes.Clone(prefix), referenceKT128(tc.message, tc.custom, Size)...)
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
			wantAfterWrite := referenceKT128(append(bytes.Clone(tc.message), "tail"...), tc.custom, Size)
			if next := h.Sum(nil); !bytes.Equal(next, wantAfterWrite) {
				t.Fatalf("Sum() after Write = %x, want %x", next, wantAfterWrite)
			}
		})
	}
}

func TestHasherSumPanicsAfterRead(t *testing.T) {
	for _, readLen := range []int{0, 1, Size} {
		t.Run(fmt.Sprintf("read=%d", readLen), func(t *testing.T) {
			h := New(nil)
			_, _ = h.Write([]byte("message"))
			_, _ = h.Read(make([]byte, readLen))
			mustPanic(t, "kt128: Hasher is finalized", func() {
				h.Sum(nil)
			})
		})
	}
}

func TestHasherSize(t *testing.T) {
	if got := New(nil).Size(); got != Size {
		t.Fatalf("Size() = %d, want %d", got, Size)
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

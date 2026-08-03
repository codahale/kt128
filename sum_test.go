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

func TestSum256(t *testing.T) {
	tests := []struct {
		messageLen int
		custom     []byte
	}{
		{messageLen: 0},
		{messageLen: 1},
		{messageLen: ChunkSize - 1},
		{messageLen: ChunkSize},
		{messageLen: ChunkSize + 1},
		{messageLen: 9 * ChunkSize},
		{messageLen: ChunkSize - 4, custom: []byte("domain")},
		{messageLen: ChunkSize, custom: ptn(41)},
	}

	for _, tc := range tests {
		name := fmt.Sprintf("message=%d/custom=%d", tc.messageLen, len(tc.custom))
		t.Run(name, func(t *testing.T) {
			message := ptn(tc.messageLen)
			want := referenceKT128(message, tc.custom, 32)
			got := Sum256(message, tc.custom)
			if !bytes.Equal(got[:], want) {
				t.Fatalf("Sum256() = %x, want %x", got, want)
			}
		})
	}
}

func TestSum256DoesNotModifyInputs(t *testing.T) {
	message := ptn(ChunkSize + 1)
	customization := ptn(41)
	wantMessage := bytes.Clone(message)
	wantCustomization := bytes.Clone(customization)

	_ = Sum256(message, customization)

	if !bytes.Equal(message, wantMessage) {
		t.Fatal("Sum256 modified its message input")
	}
	if !bytes.Equal(customization, wantCustomization) {
		t.Fatal("Sum256 modified its customization input")
	}
}

func TestSum256RFCVector(t *testing.T) {
	want := mustHex("1AC2D450FC3B4205D19DA7BFCA1B37513C0803577AC7167F06FE2CE1F0EF39E5")
	got := Sum256(nil, nil)
	if !bytes.Equal(got[:], want) {
		t.Fatalf("Sum256(nil, nil) = %x, want %x", got, want)
	}
}

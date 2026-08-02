package kt128

import (
	"bytes"
	"fmt"
	"testing"
)

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

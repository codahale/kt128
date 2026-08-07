package kt128

import (
	"bytes"
	"fmt"
	"testing"
)

func TestLengthEncode(t *testing.T) {
	tests := []struct {
		value uint64
		want  []byte
	}{
		{0, []byte{0x00}}, // RFC 9861 example
		{1, []byte{0x01, 0x01}},
		{12, []byte{0x0C, 0x01}}, // RFC 9861 example
		{127, []byte{0x7F, 0x01}},
		{128, []byte{0x80, 0x01}},
		{255, []byte{0xFF, 0x01}},
		{256, []byte{0x01, 0x00, 0x02}},
		{65535, []byte{0xFF, 0xFF, 0x02}},
		{65536, []byte{0x01, 0x00, 0x00, 0x03}},
		{65538, []byte{0x01, 0x00, 0x02, 0x03}}, // RFC 9861 example
		{1<<24 - 1, []byte{0xFF, 0xFF, 0xFF, 0x03}},
		{1 << 24, []byte{0x01, 0x00, 0x00, 0x00, 0x04}},
		{1<<32 - 1, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0x04}},
		{1 << 32, []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x05}},
		{1 << 40, []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06}},
		{1 << 48, []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07}},
		{1 << 56, []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08}},
		{^uint64(0), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x08}},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%d", tc.value), func(t *testing.T) {
			if got := lengthEncode(nil, tc.value); !bytes.Equal(got, tc.want) {
				t.Errorf("lengthEncode(nil, %d) = %x, want %x", tc.value, got, tc.want)
			}

			// lengthEncode appends to its destination, so a non-empty prefix must
			// survive.
			prefix := []byte{0xAA, 0xBB}
			want := append([]byte{0xAA, 0xBB}, tc.want...)
			if got := lengthEncode(prefix, tc.value); !bytes.Equal(got, want) {
				t.Errorf("lengthEncode(prefix, %d) = %x, want %x", tc.value, got, want)
			}
		})
	}
}

func TestTreeLeafCount(t *testing.T) {
	tests := []struct {
		message uint64
		suffix  uint64
		want    uint64
	}{
		{ChunkSize, 1, 1},
		{0, ChunkSize + 1, 1},
		{2 * ChunkSize, 0, 1},
		{2 * ChunkSize, 1, 2},
		{^uint64(0), 1, 1<<51 - 1},
		{^uint64(0), ChunkSize + 1, 1 << 51},
	}

	for _, tc := range tests {
		if got := treeLeafCount(tc.message, tc.suffix); got != tc.want {
			t.Errorf("treeLeafCount(%d, %d) = %d, want %d", tc.message, tc.suffix, got, tc.want)
		}
	}
}

//go:build arm64 && !purego

package kt128

import (
	"bytes"
	"testing"
)

func TestARM64DirectFlushChunks(t *testing.T) {
	for n, want := range map[int]int{
		2:  2,
		3:  2,
		4:  4,
		5:  5,
		7:  7,
		9:  9,
		11: 10,
		13: 12,
		15: 15,
	} {
		if got := directFlushChunks(n); got != want {
			t.Errorf("directFlushChunks(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestARM64DirectWriteUsesBatch5(t *testing.T) {
	msg := ptn(7 * BlockSize)
	h := New(nil)
	_, _ = h.Write(msg[:2*BlockSize])
	_, _ = h.Write(msg[2*BlockSize:])

	if h.leafCount != 6 {
		t.Fatalf("leaf count = %d, want 6", h.leafCount)
	}
	if cap(h.buf) != 0 {
		t.Fatalf("buffer capacity = %d, want 0", cap(h.buf))
	}

	got := make([]byte, 32)
	_, _ = h.Read(got)
	if want := referenceKT128(msg, nil, len(got)); !bytes.Equal(got, want) {
		t.Fatalf("output = %x, want %x", got, want)
	}
}

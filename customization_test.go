package kt128

import (
	"bytes"
	"testing"
)

func TestNewXOFCopiesCustomization(t *testing.T) {
	source := ptn(41)
	want := bytes.Clone(source)
	h := NewXOF(source)

	clear(source)
	if !bytes.Equal(h.c, want) {
		t.Fatal("NewXOF did not copy its customization input")
	}

	var got [32]byte
	_, _ = h.Read(got[:])
	if expected := referenceKT128(nil, want, len(got)); !bytes.Equal(got[:], expected) {
		t.Fatal("mutating the constructor input changed the hash output")
	}
}

func TestNewHashCopiesCustomization(t *testing.T) {
	source := ptn(41)
	want := bytes.Clone(source)
	h := NewHash(source)

	clear(source)
	if !bytes.Equal(h.c, want) {
		t.Fatal("NewHash did not copy its customization input")
	}
	if got, expected := h.Sum(nil), referenceKT128(nil, want, h.Size()); !bytes.Equal(got, expected) {
		t.Fatal("mutating the constructor input changed the hash digest")
	}
}

func TestCloneCopiesCustomization(t *testing.T) {
	custom := []byte("domain")
	h := NewXOF(custom)
	clone := h.Clone()
	clone.c[0] ^= 0xFF
	if !bytes.Equal(h.c, custom) {
		t.Fatalf("modifying clone customization changed original: %x", h.c)
	}
}

package kt128

import (
	"bytes"
	"testing"
)

func TestNewCopiesCustomization(t *testing.T) {
	source := ptn(41)
	want := bytes.Clone(source)
	h := New(source)

	clear(source)
	if !bytes.Equal(h.c, want) {
		t.Fatal("New did not copy its customization input")
	}

	var got [32]byte
	_, _ = h.Read(got[:])
	if expected := referenceKT128(nil, want, len(got)); !bytes.Equal(got[:], expected) {
		t.Fatal("mutating the constructor input changed the hash output")
	}
}

func TestCloneCopiesCustomization(t *testing.T) {
	custom := []byte("domain")
	h := New(custom)
	cloner, err := h.Clone()
	if err != nil {
		t.Fatalf("Clone: %v", err)
	}
	clone := cloner.(*Hasher)
	clone.c[0] ^= 0xFF
	if !bytes.Equal(h.c, custom) {
		t.Fatalf("modifying clone customization changed original: %x", h.c)
	}
}

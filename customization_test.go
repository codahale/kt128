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

func TestClear(t *testing.T) {
	h := New(ptn(41))
	customStorage := h.c
	_, _ = h.Write(ptn(ChunkSize + 1))

	h.Clear()
	if h.state != stateCleared || h.c != nil || h.pos != 0 || h.leafLen != 0 {
		t.Fatalf("Clear did not invalidate Hasher: %#v", h)
	}
	if h.final != (sponge{}) || h.leaf != (sponge{}) {
		t.Fatalf("Clear retained sponge state: %#v", h)
	}
	if bytes.Count(customStorage, []byte{0}) != len(customStorage) {
		t.Fatalf("Clear did not overwrite customization storage: %x", customStorage)
	}

	// Clear is idempotent, including on a nil receiver.
	h.Clear()
	var nilHasher *Hasher
	nilHasher.Clear()
}

func TestClearInvalidatesHasher(t *testing.T) {
	operations := []struct {
		name string
		fn   func(*Hasher)
	}{
		{"Write", func(h *Hasher) { _, _ = h.Write([]byte("x")) }},
		{"Read", func(h *Hasher) { _, _ = h.Read(make([]byte, 32)) }},
		{"Reset", func(h *Hasher) { h.Reset() }},
		{"Clone", func(h *Hasher) { _ = h.Clone() }},
		{"Pos", func(h *Hasher) { _ = h.Pos() }},
	}
	for _, op := range operations {
		t.Run(op.name, func(t *testing.T) {
			h := New([]byte("domain"))
			h.Clear()
			mustPanic(t, "kt128: Hasher is cleared", func() { op.fn(h) })
		})
	}

	h := New(nil)
	h.Clear()
	if got := h.BlockSize(); got != rate {
		t.Fatalf("BlockSize() after Clear = %d, want %d", got, rate)
	}
}

func TestClearFinalizedHasher(t *testing.T) {
	h := New([]byte("domain"))
	_, _ = h.Read(make([]byte, 32))
	if h.state != stateFinalized {
		t.Fatal("Read did not finalize Hasher")
	}

	h.Clear()
	if h.state != stateCleared || h.final != (sponge{}) {
		t.Fatalf("Clear did not wipe finalized Hasher: %#v", h)
	}
	mustPanic(t, "kt128: Hasher is cleared", func() {
		_, _ = h.Read(make([]byte, 32))
	})
}

func TestCloneCopiesCustomization(t *testing.T) {
	custom := []byte("domain")
	h := New(custom)
	clone := h.Clone()
	originalStorage := h.c
	cloneStorage := clone.c

	h.Clear()
	if bytes.Count(originalStorage, []byte{0}) != len(originalStorage) {
		t.Fatalf("clearing original did not overwrite its customization: %x", originalStorage)
	}
	if !bytes.Equal(cloneStorage, custom) {
		t.Fatalf("clearing original changed clone customization: %x", cloneStorage)
	}

	var got [64]byte
	_, _ = clone.Read(got[:])
	if want := referenceKT128(nil, custom, len(got)); !bytes.Equal(got[:], want) {
		t.Fatal("clearing original changed clone output")
	}

	clone.Clear()
	if bytes.Count(cloneStorage, []byte{0}) != len(cloneStorage) {
		t.Fatalf("clearing clone did not overwrite its customization: %x", cloneStorage)
	}
}

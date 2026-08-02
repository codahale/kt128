package kt128

import (
	"bytes"
	"testing"
)

func TestNewCustomizationStringCopiesInput(t *testing.T) {
	source := ptn(41)
	want := bytes.Clone(source)
	custom := NewCustomizationString(source)

	clear(source)
	if !bytes.Equal(custom.data, want) {
		t.Fatal("NewCustomizationString did not copy its input")
	}

	h := New(custom)
	var got [32]byte
	_, _ = h.Read(got[:])
	if expected := referenceKT128(nil, want, len(got)); !bytes.Equal(got[:], expected) {
		t.Fatal("mutating the constructor input changed the hash output")
	}
}

func TestCustomizationStringClear(t *testing.T) {
	custom := NewCustomizationString(ptn(41))
	storage := custom.data

	custom.Clear()
	if !custom.cleared || custom.data != nil {
		t.Fatalf("Clear did not invalidate customization string: %#v", custom)
	}
	if bytes.Count(storage, []byte{0}) != len(storage) {
		t.Fatalf("Clear did not overwrite customization storage: %x", storage)
	}

	// Clear is idempotent, including on a nil receiver.
	custom.Clear()
	var nilCustom *CustomizationString
	nilCustom.Clear()
}

func TestClearedCustomizationStringPanicsOnFinalization(t *testing.T) {
	custom := NewCustomizationString([]byte("domain"))
	h := New(custom)
	_, _ = h.Write([]byte("message"))
	custom.Clear()

	mustPanic(t, "kt128: CustomizationString is cleared", func() {
		_, _ = h.Read(make([]byte, 32))
	})
	if h.state == stateFinalized {
		t.Fatal("failed finalization marked Hasher finalized")
	}
}

func TestFinalizedHasherSurvivesCustomizationStringClear(t *testing.T) {
	custom := NewCustomizationString([]byte("domain"))
	h := New(custom)
	first := make([]byte, 32)
	_, _ = h.Read(first)

	custom.Clear()
	second := make([]byte, 32)
	_, _ = h.Read(second)

	want := referenceKT128(nil, []byte("domain"), 64)
	if !bytes.Equal(first, want[:32]) || !bytes.Equal(second, want[32:]) {
		t.Fatal("clearing customization changed an already-finalized Hasher")
	}
}

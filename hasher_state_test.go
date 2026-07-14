package kt128

import (
	"bytes"
	"fmt"
	"testing"
)

func TestClone(t *testing.T) {
	sizes := []int{0, 1, BlockSize - 1, BlockSize, BlockSize + 1, 83521}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%d", size), func(t *testing.T) {
			msg := ptn(size)

			// Write all data, clone, verify both produce the same output.
			h := New(nil)
			_, _ = h.Write(msg)

			clone := h.Clone()

			// Finalizing the original must not affect the clone.
			want := make([]byte, 64)
			_, _ = h.Read(want)

			got := make([]byte, 64)
			_, _ = clone.Read(got)

			if !bytes.Equal(got, want) {
				t.Errorf("size=%d: clone output mismatch", size)
			}
		})
	}

	t.Run("independent after clone", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(BlockSize + 1))

		clone := h.Clone()

		// Write more data to the original only.
		_, _ = h.Write([]byte("extra"))

		out1 := make([]byte, 64)
		_, _ = h.Read(out1)

		out2 := make([]byte, 64)
		_, _ = clone.Read(out2)

		if bytes.Equal(out1, out2) {
			t.Error("clone and original produced identical output after diverging")
		}
	})
}

func TestEqual(t *testing.T) {
	t.Run("same input", func(t *testing.T) {
		h1 := New(nil)
		_, _ = h1.Write(ptn(100))

		h2 := New(nil)
		_, _ = h2.Write(ptn(100))

		if h1.Equal(h2) != 1 {
			t.Fatal("identical hashers should be equal")
		}
	})

	t.Run("different input", func(t *testing.T) {
		h1 := New(nil)
		_, _ = h1.Write(ptn(100))

		h2 := New(nil)
		_, _ = h2.Write(ptn(200))

		if h1.Equal(h2) != 0 {
			t.Fatal("different hashers should not be equal")
		}
	})

	t.Run("clone", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(BlockSize + 1))

		clone := h.Clone()
		if h.Equal(clone) != 1 {
			t.Fatal("hasher and its clone should be equal")
		}
	})

	t.Run("diverged clone", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(100))

		clone := h.Clone()
		_, _ = clone.Write([]byte("extra"))

		if h.Equal(clone) != 0 {
			t.Fatal("diverged clone should not be equal")
		}
	})

	t.Run("different customization", func(t *testing.T) {
		// The customization string is only absorbed at finalization, but it
		// still distinguishes two otherwise-identical hashers: they would
		// produce different output, so they must not compare equal.
		h1 := New([]byte("alpha"))
		_, _ = h1.Write(ptn(100))

		h2 := New([]byte("beta"))
		_, _ = h2.Write(ptn(100))

		if h1.Equal(h2) != 0 {
			t.Fatal("hashers with different customization strings should not be equal")
		}
	})

	t.Run("same customization", func(t *testing.T) {
		h1 := New([]byte("alpha"))
		_, _ = h1.Write(ptn(100))

		h2 := New([]byte("alpha"))
		_, _ = h2.Write(ptn(100))

		if h1.Equal(h2) != 1 {
			t.Fatal("hashers with the same customization string should be equal")
		}
	})
}

func TestPos(t *testing.T) {
	t.Run("new hasher", func(t *testing.T) {
		h := New(nil)
		if h.Pos() != 0 {
			t.Fatalf("Pos() = %d, want 0", h.Pos())
		}
	})

	t.Run("after write", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(100))
		if h.Pos() != 100 {
			t.Fatalf("Pos() = %d, want 100", h.Pos())
		}
	})

	t.Run("cumulative writes", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(100))
		_, _ = h.Write(ptn(200))
		if h.Pos() != 300 {
			t.Fatalf("Pos() = %d, want 300", h.Pos())
		}
	})

	t.Run("after reset", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(100))
		h.Reset()
		if h.Pos() != 0 {
			t.Fatalf("Pos() after Reset = %d, want 0", h.Pos())
		}
	})
}

func TestReset(t *testing.T) {
	h := New(nil)
	_, _ = h.Write(ptn(BlockSize + 1))
	h.Reset()
	_, _ = h.Write(ptn(BlockSize + 1))

	fresh := New(nil)
	_, _ = fresh.Write(ptn(BlockSize + 1))

	out1 := make([]byte, 64)
	_, _ = h.Read(out1)

	out2 := make([]byte, 64)
	_, _ = fresh.Read(out2)

	if !bytes.Equal(out1, out2) {
		t.Fatal("Reset hasher should produce same output as fresh hasher")
	}
}

func TestClear(t *testing.T) {
	custom := ptn(41)
	h := New(custom)

	// Fill the entire buffer allocation, including capacity beyond its length,
	// so Clear must scrub storage that may retain bytes from earlier writes.
	h.buf = make([]byte, BlockSize, BlockSize+257)
	buffer := h.buf[:cap(h.buf)]
	for i := range buffer {
		buffer[i] = 0xA5
	}
	h.buf = h.buf[:0]
	h.final.a[0] = 1
	h.final.a[lanes-1] = 2
	h.final.pos = 17
	h.pos = BlockSize
	h.leafCount = 3
	h.pendingLen = rate
	h.state = stateFinalized
	h.ds = treeDS
	final := &h.final

	h.Clear()

	for i, b := range buffer {
		if b != 0 {
			t.Fatalf("buffer[%d] = %#x after Clear, want 0", i, b)
		}
	}
	if *final != (sponge{}) {
		t.Fatalf("final sponge after Clear = %#v, want zero", *final)
	}
	if h.buf != nil {
		t.Fatalf("buffer after Clear = %#v, want nil", h.buf)
	}
	if h.pos != 0 || h.leafCount != 0 || h.pendingLen != 0 || h.state != stateSingle || h.ds != 0 {
		t.Fatalf("Hasher metadata was not reset by Clear: %#v", h)
	}
	if !bytes.Equal(h.c, custom) {
		t.Fatal("Clear did not preserve the customization string")
	}

	msg := ptn(BlockSize + 1)
	_, _ = h.Write(msg)
	got := make([]byte, 64)
	_, _ = h.Read(got)

	fresh := New(custom)
	_, _ = fresh.Write(msg)
	want := make([]byte, 64)
	_, _ = fresh.Read(want)
	if !bytes.Equal(got, want) {
		t.Fatal("Hasher reused after Clear differs from a fresh customized Hasher")
	}
}

// TestCustomizationStringCopied verifies that New copies the customization
// string, so mutating the caller's slice afterward cannot change the output.
func TestCustomizationStringCopied(t *testing.T) {
	msg := ptn(100)
	custom := ptn(41) // caller-owned, mutable

	// Reference output for the original customization.
	want := make([]byte, 64)
	ref := New(custom)
	_, _ = ref.Write(msg)
	_, _ = ref.Read(want)

	// Mutate the caller's slice after constructing the hasher.
	h := New(custom)
	for i := range custom {
		custom[i] ^= 0xFF
	}
	_, _ = h.Write(msg)
	got := make([]byte, 64)
	_, _ = h.Read(got)

	if !bytes.Equal(got, want) {
		t.Fatal("mutating the caller's customization slice changed the output; New did not copy it")
	}
}

// TestResetPreservesCustomization verifies that Reset keeps the customization
// string passed to New, so a reused hasher matches a fresh one constructed with
// the same customization.
func TestResetPreservesCustomization(t *testing.T) {
	custom := ptn(41)

	h := New(custom)
	_, _ = h.Write(ptn(100))
	h.Reset()
	_, _ = h.Write(ptn(200))

	fresh := New(custom)
	_, _ = fresh.Write(ptn(200))

	out1 := make([]byte, 64)
	_, _ = h.Read(out1)
	out2 := make([]byte, 64)
	_, _ = fresh.Read(out2)

	if !bytes.Equal(out1, out2) {
		t.Fatal("Reset should preserve the customization string")
	}
}

func TestBlockSizeMethod(t *testing.T) {
	h := New(nil)
	if got := h.BlockSize(); got != BlockSize {
		t.Fatalf("BlockSize() = %d, want %d", got, BlockSize)
	}
}

// TestLengthEncode checks lengthEncode against hand-computed golden values for
// RFC 9861 §2.3.1's own examples and every byte-width boundary. The leafCount and
// |C| encodings only reach the multi-byte forms on very large inputs, so the
// RFC vectors and the (size-capped) fuzzer barely exercise them; these golden
// values pin the encoding independently of any other implementation in the tree.

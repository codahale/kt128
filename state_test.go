package kt128

import (
	"bytes"
	"fmt"
	"testing"
)

func TestXOFClone(t *testing.T) {
	sizes := []int{0, 1, ChunkSize - 1, ChunkSize, ChunkSize + 1, 83521}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%d", size), func(t *testing.T) {
			msg := ptn(size)

			// Write all data, clone, verify both produce the same output.
			h := NewXOF(nil)
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
		h := NewXOF(nil)
		_, _ = h.Write(ptn(ChunkSize + 1))

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

func TestXOFZeroValue(t *testing.T) {
	var x XOF
	message := []byte("message")
	_, _ = x.Write(message)
	got := make([]byte, 64)
	_, _ = x.Read(got)
	if want := referenceKT128(message, nil, len(got)); !bytes.Equal(got, want) {
		t.Fatalf("zero-value output = %x, want %x", got, want)
	}
}

func TestHashClone(t *testing.T) {
	custom := []byte("domain")
	h := NewHash(custom, 57)
	_, _ = h.Write(ptn(ChunkSize + 1))

	cloner, err := h.Clone()
	if err != nil {
		t.Fatal(err)
	}
	clone := cloner.(*Hash)
	if clone.Size() != h.Size() {
		t.Fatalf("clone Size() = %d, want %d", clone.Size(), h.Size())
	}
	clone.c[0] ^= 0xff
	if !bytes.Equal(h.c, custom) {
		t.Fatal("clone retained the original customization storage")
	}

	want := h.Sum(nil)
	if got := clone.Sum(nil); bytes.Equal(got, want) {
		t.Fatal("mutating clone customization did not change its digest")
	}
	clone = cloner.(*Hash)
	clone.c[0] ^= 0xff
	if got := clone.Sum(nil); !bytes.Equal(got, want) {
		t.Fatal("restored clone does not match original")
	}
}

func TestPos(t *testing.T) {
	t.Run("new XOF", func(t *testing.T) {
		h := NewXOF(nil)
		if h.Pos() != 0 {
			t.Fatalf("Pos() = %d, want 0", h.Pos())
		}
	})

	t.Run("after write", func(t *testing.T) {
		h := NewXOF(nil)
		_, _ = h.Write(ptn(100))
		if h.Pos() != 100 {
			t.Fatalf("Pos() = %d, want 100", h.Pos())
		}
	})

	t.Run("cumulative writes", func(t *testing.T) {
		h := NewXOF(nil)
		_, _ = h.Write(ptn(100))
		_, _ = h.Write(ptn(200))
		if h.Pos() != 300 {
			t.Fatalf("Pos() = %d, want 300", h.Pos())
		}
	})

	t.Run("after reset", func(t *testing.T) {
		h := NewXOF(nil)
		_, _ = h.Write(ptn(100))
		h.Reset()
		if h.Pos() != 0 {
			t.Fatalf("Pos() after Reset = %d, want 0", h.Pos())
		}
	})
}

func TestReset(t *testing.T) {
	h := NewXOF(nil)
	_, _ = h.Write(ptn(ChunkSize + 1))
	h.Reset()
	if h.final != (sponge{}) || h.leaf != (sponge{}) || h.leafLen != 0 {
		t.Fatalf("Reset retained sponge state: %#v", h)
	}
	_, _ = h.Write(ptn(ChunkSize + 1))

	fresh := NewXOF(nil)
	_, _ = fresh.Write(ptn(ChunkSize + 1))

	out1 := make([]byte, 64)
	_, _ = h.Read(out1)

	out2 := make([]byte, 64)
	_, _ = fresh.Read(out2)

	if !bytes.Equal(out1, out2) {
		t.Fatal("Reset XOF should produce same output as fresh XOF")
	}
}

// TestResetPreservesCustomization verifies that Reset keeps the customization
// string passed to NewXOF, so a reused XOF matches a fresh one constructed with
// the same customization.
func TestResetPreservesCustomization(t *testing.T) {
	custom := ptn(41)

	h := NewXOF(custom)
	_, _ = h.Write(ptn(100))
	h.Reset()
	_, _ = h.Write(ptn(200))

	fresh := NewXOF(custom)
	_, _ = fresh.Write(ptn(200))

	out1 := make([]byte, 64)
	_, _ = h.Read(out1)
	out2 := make([]byte, 64)
	_, _ = fresh.Read(out2)

	if !bytes.Equal(out1, out2) {
		t.Fatal("Reset should preserve the customization string")
	}
}

func TestBlockAndChunkSizes(t *testing.T) {
	h := NewXOF(nil)
	if got, want := h.BlockSize(), 168; got != want {
		t.Fatalf("BlockSize() = %d, want %d", got, want)
	}
	if got, want := ChunkSize, 8192; got != want {
		t.Fatalf("ChunkSize = %d, want %d", got, want)
	}
}

// TestLengthEncode checks lengthEncode against hand-computed golden values for
// RFC 9861 §2.3.1's own examples and every byte-width boundary. The leafCount and
// |C| encodings only reach the multi-byte forms on very large inputs, so the
// RFC vectors and the (size-capped) fuzzer barely exercise them; these golden
// values pin the encoding independently of any other implementation in the tree.

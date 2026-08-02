package kt128

import (
	"bytes"
	"testing"
)

func TestWriteFusedS0Leaf(t *testing.T) {
	sizes := []int{
		2 * ChunkSize, 2*ChunkSize + 1, 3 * ChunkSize, 5*ChunkSize + 11,
		8 * ChunkSize, 8*ChunkSize + 37, 9 * ChunkSize, 16 * ChunkSize, 16*ChunkSize + 5,
	}
	for _, size := range sizes {
		msg := ptn(size)

		one := New(nil)
		_, _ = one.Write(msg)
		got := make([]byte, 64)
		_, _ = one.Read(got)

		two := New(nil)
		_, _ = two.Write(msg[:1]) // eager absorption forecloses fusion
		_, _ = two.Write(msg[1:])
		want := make([]byte, 64)
		_, _ = two.Read(want)

		if !bytes.Equal(got, want) {
			t.Errorf("size=%d: fused path diverges: got %x, want %x", size, got, want)
		}
	}
}

func TestWriteTreeModeIncrementalLeaf(t *testing.T) {
	t.Run("direct S0", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(ChunkSize + 1))

		if h.state != stateTree {
			t.Fatalf("state = %d, want stateTree", h.state)
		}
		if h.leafLen != 1 {
			t.Fatalf("partial leaf bytes = %d, want 1", h.leafLen)
		}
	})

	t.Run("no leaf below one chunk", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(ChunkSize))

		if h.state != stateSingle {
			t.Fatalf("state = %d, want stateSingle", h.state)
		}
		if h.leafLen != 0 || h.leaf != (sponge{}) {
			t.Fatalf("unexpected partial leaf state: len=%d state=%#v", h.leafLen, h.leaf)
		}
	})

	t.Run("fragment completes leaf", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(ChunkSize + 1))
		_, _ = h.Write(ptn(ChunkSize - 1))

		if h.leafLen != 0 || h.leaf != (sponge{}) {
			t.Fatalf("completed leaf was not reset: len=%d state=%#v", h.leafLen, h.leaf)
		}
	})

	t.Run("fragmented writes do not allocate", func(t *testing.T) {
		msg := ptn(2*availableLanes*ChunkSize + 123)
		var out [32]byte
		allocs := testing.AllocsPerRun(20, func() {
			h := New(nil)
			for off := 0; off < len(msg); off += 1024 {
				_, _ = h.Write(msg[off:min(off+1024, len(msg))])
			}
			_, _ = h.Read(out[:])
		})
		if allocs != 0 {
			t.Fatalf("fragmented hash allocated %.0f times, want 0", allocs)
		}
	})

	t.Run("process exact lane batch directly", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn((availableLanes + 1) * ChunkSize))

		if h.leafLen != 0 {
			t.Fatalf("partial leaf bytes = %d, want 0", h.leafLen)
		}
	})

	t.Run("ragged bulk write retains only leaf state", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(6*ChunkSize + 37))

		if h.leafLen != 37 {
			t.Fatalf("partial leaf bytes = %d, want 37", h.leafLen)
		}
	})
}

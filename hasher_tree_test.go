package kt128

import (
	"bytes"
	"runtime"
	"testing"
)

func TestWriteFusedS0Leaf(t *testing.T) {
	sizes := []int{
		2 * BlockSize, 2*BlockSize + 1, 3 * BlockSize, 5*BlockSize + 11,
		8 * BlockSize, 8*BlockSize + 37, 9 * BlockSize, 16 * BlockSize, 16*BlockSize + 5,
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

func TestWriteTreeModeBuffering(t *testing.T) {
	t.Run("direct S0", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(BlockSize + 1))

		if h.state != stateTree {
			t.Fatalf("state = %d, want stateTree", h.state)
		}
		if len(h.buf) != 1 {
			t.Fatalf("buffered bytes = %d, want 1", len(h.buf))
		}
		if cap(h.buf) >= BlockSize {
			t.Fatalf("buffer capacity = %d, want less than one block", cap(h.buf))
		}
	})

	t.Run("no buffering below one chunk", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(BlockSize))

		if h.state != stateSingle {
			t.Fatalf("state = %d, want stateSingle", h.state)
		}
		if cap(h.buf) != 0 {
			t.Fatalf("buffer capacity = %d, want 0", cap(h.buf))
		}

		_, _ = h.Write([]byte{0xA5})

		if h.state != stateTree {
			t.Fatalf("state = %d, want stateTree", h.state)
		}
		if len(h.buf) != 1 {
			t.Fatalf("buffered bytes = %d, want 1", len(h.buf))
		}
		if cap(h.buf) >= BlockSize {
			t.Fatalf("buffer capacity = %d, want less than one block", cap(h.buf))
		}
	})

	t.Run("streaming buffer settles after one growth", func(t *testing.T) {
		chunk := ptn(BlockSize)

		// A fresh hasher's first flush cycle grows the buffer a bounded
		// number of times — the initial exact-size fill, append-sized steps
		// up to growJumpMin, and one jump to the streaming high-water mark —
		// and later cycles reuse it without reallocating.
		wantMax := 3.0 // hasher + first fill + one growth
		if growJumpMin > 0 {
			wantMax = 5.0 // plus append's growth steps below the jump threshold
		}
		allocs := testing.AllocsPerRun(3, func() {
			h := New(nil)
			for range 2*streamChunks + 2 {
				_, _ = h.Write(chunk)
			}
		})
		if allocs > wantMax {
			t.Fatalf("streaming write cycle allocated %.0f times, want at most %.0f", allocs, wantMax)
		}

		h := New(nil)
		for range 2*streamChunks + 2 {
			_, _ = h.Write(chunk)
		}
		if maxCap := (streamChunks + 1) * BlockSize; cap(h.buf) > maxCap {
			t.Fatalf("buffer capacity = %d, want at most %d", cap(h.buf), maxCap)
		}
	})

	t.Run("flush exact lane batch", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn(BlockSize + 1))
		_, _ = h.Write(ptn(streamChunks*BlockSize - 1))

		if h.leafCount != uint64(streamChunks) {
			t.Fatalf("leaf count = %d, want %d", h.leafCount, streamChunks)
		}
		if len(h.buf) != 0 {
			t.Fatalf("buffered bytes = %d, want 0", len(h.buf))
		}
	})

	t.Run("buffered chunks complete lane batch", func(t *testing.T) {
		if streamChunks == 1 {
			t.Skip("scalar path has no multi-chunk batch")
		}

		msg := ptn((streamChunks + 3) * BlockSize)
		h := New(nil)
		_, _ = h.Write(msg[:2*BlockSize])
		_, _ = h.Write(msg[2*BlockSize : 3*BlockSize])
		if len(h.buf) != BlockSize {
			t.Fatalf("buffered bytes before top-up = %d, want %d", len(h.buf), BlockSize)
		}

		_, _ = h.Write(msg[3*BlockSize:])
		if h.leafCount != uint64(streamChunks+1) {
			t.Fatalf("leaf count = %d, want %d", h.leafCount, streamChunks+1)
		}
		if len(h.buf) != BlockSize {
			t.Fatalf("buffered bytes after top-up = %d, want %d", len(h.buf), BlockSize)
		}

		got := make([]byte, 32)
		_, _ = h.Read(got)
		if want := referenceKT128(msg, nil, len(got)); !bytes.Equal(got, want) {
			t.Fatalf("output = %x, want %x", got, want)
		}
	})

	t.Run("direct pairs below lane batch", func(t *testing.T) {
		if flushChunks() >= availableLanes {
			t.Skip("no sub-batch direct flushing on this platform")
		}
		if runtime.GOARCH == "amd64" {
			// amd64 S_0 fusion consumes four chunks, changing the shape
			// arithmetic below; TestWriteForceAVX2DirectFlush covers the
			// AVX2 sub-batch direct flush instead.
			t.Skip("amd64 shapes are covered by TestWriteForceAVX2DirectFlush")
		}
		h := New(nil)
		_, _ = h.Write(ptn(6*BlockSize + 37)) // S_0+leaf fused, 4 leaves in place, tail buffered

		if h.leafCount != 5 {
			t.Fatalf("leaf count = %d, want 5", h.leafCount)
		}
		if len(h.buf) != 37 {
			t.Fatalf("buffered bytes = %d, want 37", len(h.buf))
		}
		if cap(h.buf) >= BlockSize {
			t.Fatalf("buffer capacity = %d, want less than one block", cap(h.buf))
		}
	})

	t.Run("process exact lane batch directly", func(t *testing.T) {
		h := New(nil)
		_, _ = h.Write(ptn((availableLanes + 1) * BlockSize))

		if h.leafCount != uint64(availableLanes) {
			t.Fatalf("leaf count = %d, want %d", h.leafCount, availableLanes)
		}
		if cap(h.buf) != 0 {
			t.Fatalf("buffer capacity = %d, want 0", cap(h.buf))
		}
	})

	t.Run("chunk-aligned remainder drains in place", func(t *testing.T) {
		// A bulk write ending on a chunk boundary drains its sub-unit chunk
		// remainder in place: the buffer stays unallocated and every complete
		// leaf is counted. 30 chunks leaves a six-chunk aligned remainder on
		// AVX-512 (S_0 fusion takes 8) and a two-chunk one on AVX2 (fusion
		// takes 4); arm64's fusion takes 2 and its 2-chunk flush unit covers
		// the rest exactly, and purego's single-chunk unit leaves no
		// remainder — both pin the same invariant trivially.
		h := New(nil)
		_, _ = h.Write(ptn(30 * BlockSize))

		if h.leafCount != 29 {
			t.Fatalf("leaf count = %d, want 29", h.leafCount)
		}
		if cap(h.buf) != 0 {
			t.Fatalf("buffer capacity = %d, want 0", cap(h.buf))
		}
	})
}

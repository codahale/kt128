package kt128

import (
	"bufio"
	"fmt"
	"io"
	"testing"
)

func BenchmarkWrite(b *testing.B) {
	for _, size := range sizes {
		b.Run(size.Name, func(b *testing.B) {
			msg := ptn(size.N)
			out := make([]byte, 32)
			b.SetBytes(int64(size.N))
			b.ReportAllocs()
			for b.Loop() {
				h := New(nil)
				_, _ = h.Write(msg)
				_, _ = h.Read(out)
			}
		})
	}
}

func BenchmarkWriteStreaming(b *testing.B) {
	for _, size := range sizes {
		if size.N < 2*ChunkSize {
			continue
		}
		b.Run(size.Name, func(b *testing.B) {
			msg := ptn(size.N)
			out := make([]byte, 32)
			b.SetBytes(int64(size.N))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				h := New(nil)
				for i := 0; i < len(msg); i += ChunkSize {
					end := min(i+ChunkSize, len(msg))
					_, _ = h.Write(msg[i:end])
				}
				_, _ = h.Read(out)
			}
		})
	}
}

// BenchmarkWriteFragmented compares the fixed-state streaming path with an
// explicitly buffered writer. The Hasher and bufio.Writer are reused so the
// benchmark measures steady-state absorption rather than wrapper allocation.
func BenchmarkWriteFragmented(b *testing.B) {
	const size = 1024 * 1024
	msg := ptn(size)
	for _, fragment := range []int{1024, ChunkSize, 2 * ChunkSize, 4 * ChunkSize, 5 * ChunkSize, 8 * ChunkSize} {
		name := fmt.Sprintf("%dB", fragment)
		b.Run("raw/"+name, func(b *testing.B) {
			h := New(nil)
			var out [32]byte
			b.SetBytes(size)
			b.ReportAllocs()
			for b.Loop() {
				h.Reset()
				for off := 0; off < len(msg); off += fragment {
					_, _ = h.Write(msg[off:min(off+fragment, len(msg))])
				}
				_, _ = h.Read(out[:])
			}
		})
	}

	for _, bufferSize := range []int{4096, 5 * ChunkSize, 8 * ChunkSize} {
		name := fmt.Sprintf("%dB", bufferSize)
		b.Run("bufio/"+name, func(b *testing.B) {
			h := New(nil)
			w := bufio.NewWriterSize(h, bufferSize)
			var out [32]byte
			b.SetBytes(size)
			b.ReportAllocs()
			for b.Loop() {
				h.Reset()
				w.Reset(h)
				for off := 0; off < len(msg); off += 1024 {
					_, _ = w.Write(msg[off:min(off+1024, len(msg))])
				}
				_ = w.Flush()
				_, _ = h.Read(out[:])
			}
		})
	}
}

// BenchmarkRead measures steady-state squeeze throughput: the hasher is
// finalized once and each iteration continues the XOF output stream, so no
// setup or absorption is timed.
func BenchmarkRead(b *testing.B) {
	for _, outSize := range []int{32, 64, 256, 1024} {
		b.Run(fmt.Sprintf("%d", outSize), func(b *testing.B) {
			h := New(nil)
			_, _ = h.Write(ptn(ChunkSize + 1))
			out := make([]byte, outSize)
			b.SetBytes(int64(outSize))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				_, _ = io.ReadFull(h, out)
			}
		})
	}
}

type size struct {
	Name string
	N    int
}

var sizes = []size{
	{"1B", 1},
	{"64B", 64},
	{"8KiB", 8 * 1024},
	{"8KiB+1B", ChunkSize + 1},
	{"16KiB", 16 * 1024},
	{"28KiB", 7 * ChunkSize / 2},
	{"32KiB", 32 * 1024},
	{"64KiB", 64 * 1024},
	{"72KiB", 9 * ChunkSize},
	{"1MiB", 1024 * 1024},
	{"16MiB", 16 * 1024 * 1024},
}

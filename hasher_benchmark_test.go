package kt128

import (
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
		if size.N < 2*BlockSize {
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
				for i := 0; i < len(msg); i += BlockSize {
					end := min(i+BlockSize, len(msg))
					_, _ = h.Write(msg[i:end])
				}
				_, _ = h.Read(out)
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
			_, _ = h.Write(ptn(BlockSize + 1))
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
	{"8KiB+1B", BlockSize + 1},
	{"16KiB", 16 * 1024},
	{"28KiB", 7 * BlockSize / 2},
	{"32KiB", 32 * 1024},
	{"64KiB", 64 * 1024},
	{"72KiB", 9 * BlockSize},
	{"1MiB", 1024 * 1024},
	{"16MiB", 16 * 1024 * 1024},
}

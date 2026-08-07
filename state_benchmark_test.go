package kt128

import (
	"bufio"
	"fmt"
	"testing"
)

func BenchmarkHasherWrite(b *testing.B) {
	for _, tc := range []struct {
		name string
		size int
	}{
		{"64B", 64},
		{"8KiB", ChunkSize},
		{"64KiB", 8 * ChunkSize},
		{"1MiB", 1024 * 1024},
	} {
		b.Run("one-shot/"+tc.name, func(b *testing.B) {
			benchmarkWrites(b, ptn(tc.size), tc.size)
		})
	}

	const messageSize = 1024 * 1024
	msg := ptn(messageSize)
	for _, writeSize := range []int{
		1024,
		ChunkSize,
		RecommendedWriteBufferSize(),
	} {
		name := fmt.Sprintf("direct/1MiB/%dKiB-writes", writeSize/1024)
		b.Run(name, func(b *testing.B) {
			benchmarkWrites(b, msg, writeSize)
		})
	}

	recommended := RecommendedWriteBufferSize()
	for _, bufferSize := range []int{
		4096,
		ChunkSize,
		2 * ChunkSize,
		4 * ChunkSize,
		5 * ChunkSize,
		8 * ChunkSize,
		16 * ChunkSize,
	} {
		name := fmt.Sprintf("bufio/1MiB/1KiB-writes/%dKiB-buffer", bufferSize/1024)
		if bufferSize == recommended {
			name += "-recommended"
		}
		b.Run(name, func(b *testing.B) {
			h := NewXOF(nil)
			w := bufio.NewWriterSize(h, bufferSize)
			var out [32]byte
			b.SetBytes(messageSize)
			b.ReportAllocs()
			b.ResetTimer()
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

func benchmarkWrites(b *testing.B, msg []byte, writeSize int) {
	h := NewXOF(nil)
	var out [32]byte
	b.Helper()
	b.SetBytes(int64(len(msg)))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		h.Reset()
		for off := 0; off < len(msg); off += writeSize {
			_, _ = h.Write(msg[off:min(off+writeSize, len(msg))])
		}
		_, _ = h.Read(out[:])
	}
}

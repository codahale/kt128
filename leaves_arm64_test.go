//go:build arm64 && !purego

package kt128

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

func TestWriteForceGenericFallback(t *testing.T) {
	savedSHA3 := cpuid.HasSHA3
	savedHasLeafBatch5 := hasLeafBatch5
	defer func() {
		cpuid.HasSHA3 = savedSHA3
		hasLeafBatch5 = savedHasLeafBatch5
	}()
	cpuid.HasSHA3 = false
	hasLeafBatch5 = false

	for _, size := range []int{
		0, ChunkSize, 2 * ChunkSize, 9*ChunkSize + 137, 1024 * 1024,
	} {
		t.Run(fmt.Sprintf("%d", size), func(t *testing.T) {
			msg := ptn(size)
			h := New([]byte("generic-fallback"))
			for off := 0; off < len(msg); off += 3333 {
				_, _ = h.Write(msg[off:min(off+3333, len(msg))])
			}
			got := make([]byte, 64)
			_, _ = h.Read(got)
			want := referenceKT128(msg, []byte("generic-fallback"), len(got))
			if !bytes.Equal(got, want) {
				t.Fatalf("generic fallback output %x != reference %x", got, want)
			}
		})
	}
}

func TestARM64TripleTailScheduling(t *testing.T) {
	if !cpuid.HasSHA3 {
		t.Skip("no SHA3 extension")
	}
	if got := fuseS0Chunks(3, tripleSerialTailBlocks*rate-1); got != 3 {
		t.Fatalf("fuseS0Chunks below crossover = %d, want 3", got)
	}
	if got := fuseS0Chunks(3, tripleSerialTailBlocks*rate); got != 2 {
		t.Fatalf("fuseS0Chunks at crossover = %d, want 2", got)
	}
	if got := fuseTailChunks(3, tripleSerialTailBlocks-1); got != 0 {
		t.Fatalf("fuseTailChunks below crossover = %d, want 0", got)
	}
	if got := fuseTailChunks(3, tripleSerialTailBlocks); got != 1 {
		t.Fatalf("fuseTailChunks at crossover = %d, want 1", got)
	}
}

func TestARM64DirectWriteUsesBatch5(t *testing.T) {
	if !cpuid.HasSHA3 {
		t.Skip("no SHA3 extension")
	}
	msg := ptn(7 * ChunkSize)
	h := New(nil)
	_, _ = h.Write(msg[:2*ChunkSize])
	_, _ = h.Write(msg[2*ChunkSize:])

	if h.leafLen != 0 {
		t.Fatalf("partial leaf bytes = %d, want 0", h.leafLen)
	}

	got := make([]byte, 32)
	_, _ = h.Read(got)
	if want := referenceKT128(msg, nil, len(got)); !bytes.Equal(got, want) {
		t.Fatalf("output = %x, want %x", got, want)
	}
}

func BenchmarkARM64TripleVsPairScalar(b *testing.B) {
	if !cpuid.HasSHA3 {
		b.Skip("no SHA3 extension")
	}
	input := ptn(3 * ChunkSize)
	var cvs [256]byte

	b.Run("hybrid", func(b *testing.B) {
		b.SetBytes(3 * ChunkSize)
		for b.Loop() {
			processLeavesTripleArch(input, &cvs)
		}
	})

	b.Run("pair_scalar", func(b *testing.B) {
		b.SetBytes(3 * ChunkSize)
		for b.Loop() {
			processLeavesPairARM64(&input[0], &cvs[0])
			var scalar sponge
			leafStateX1(input[2*ChunkSize:], &scalar)
			scalar.squeeze(cvs[64:96])
		}
	})
}

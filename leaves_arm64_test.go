//go:build arm64 && !purego

package kt128

import (
	"bytes"
	"testing"
)

func TestARM64DirectFlushChunks(t *testing.T) {
	for n, want := range map[int]int{
		2:  2,
		3:  3,
		4:  4,
		5:  5,
		7:  7,
		9:  9,
		11: 10,
		13: 13,
		15: 15,
	} {
		if got := directFlushChunks(n); got != want {
			t.Errorf("directFlushChunks(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestARM64TripleTailScheduling(t *testing.T) {
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
	msg := ptn(7 * BlockSize)
	h := New(nil)
	_, _ = h.Write(msg[:2*BlockSize])
	_, _ = h.Write(msg[2*BlockSize:])

	if h.leafCount != 6 {
		t.Fatalf("leaf count = %d, want 6", h.leafCount)
	}
	if cap(h.buf) != 0 {
		t.Fatalf("buffer capacity = %d, want 0", cap(h.buf))
	}

	got := make([]byte, 32)
	_, _ = h.Read(got)
	if want := referenceKT128(msg, nil, len(got)); !bytes.Equal(got, want) {
		t.Fatalf("output = %x, want %x", got, want)
	}
}

func BenchmarkARM64TripleVsPairScalar(b *testing.B) {
	input := ptn(3 * BlockSize)
	var cvs [256]byte

	b.Run("hybrid", func(b *testing.B) {
		b.SetBytes(3 * BlockSize)
		for b.Loop() {
			processLeavesTripleArch(input, &cvs)
		}
	})

	b.Run("pair_scalar", func(b *testing.B) {
		b.SetBytes(3 * BlockSize)
		for b.Loop() {
			processLeavesPairARM64(&input[0], &cvs[0])
			var scalar sponge
			leafStateX1(input[2*BlockSize:], &scalar)
			scalar.squeeze(cvs[64:96])
		}
	})
}

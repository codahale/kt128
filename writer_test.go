package kt128

import "testing"

func TestRecommendedWriteBufferSize(t *testing.T) {
	size := RecommendedWriteBufferSize()
	if size < ChunkSize {
		t.Fatalf("RecommendedWriteBufferSize() = %d, want at least %d", size, ChunkSize)
	}
	if size%ChunkSize != 0 {
		t.Fatalf("RecommendedWriteBufferSize() = %d, want a multiple of %d", size, ChunkSize)
	}
}

package kt128

// RecommendedWriteBufferSize returns a buffer size that coalesces small writes
// into batches suited to the fastest leaf implementation enabled at runtime.
// The result is always a positive multiple of [ChunkSize].
func RecommendedWriteBufferSize() int {
	return recommendedWriteBufferChunks() * ChunkSize
}

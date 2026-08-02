package kt128

// Sum256 returns the 32-byte KT128 hash of message using customization as the
// customization string. Sum256 does not retain or modify either input slice.
func Sum256(message, customization []byte) (sum [32]byte) {
	h := New(customization)
	_, _ = h.Write(message)
	_, _ = h.Read(sum[:])
	h.Clear()
	return sum
}

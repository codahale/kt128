package kt128

// Sum appends the current 32-byte KT128 digest to b without changing h's
// state. Sum panics after any call to Read, including a zero-length call.
func (h *Hasher) Sum(b []byte) []byte {
	if h.state == stateFinalized {
		panic("kt128: Hasher is finalized")
	}

	var sum [Size]byte
	clone, _ := h.Clone()
	_, _ = clone.(*Hasher).Read(sum[:])
	return append(b, sum[:]...)
}

// Sum returns outputLen bytes of the KT128 hash of message using customization
// as the customization string. Sum does not retain or modify either input
// slice. Sum panics if outputLen is negative.
func Sum(message, customization []byte, outputLen int) []byte {
	if outputLen < 0 {
		panic("kt128: negative output length")
	}

	sum := make([]byte, outputLen)
	h := New(customization)
	_, _ = h.Write(message)
	_, _ = h.Read(sum)
	return sum
}

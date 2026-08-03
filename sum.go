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

// Sum256 returns the 32-byte KT128 hash of message using customization as the
// customization string. Sum256 does not retain or modify either input slice.
func Sum256(message, customization []byte) (sum [Size]byte) {
	h := New(customization)
	_, _ = h.Write(message)
	_, _ = h.Read(sum[:])
	return sum
}

package kt128

// Clone returns an independent copy of h at its current absorption or squeeze
// position, including a copy of its customization string.
func (h *Hasher) Clone() *Hasher {
	return &Hasher{
		c:       append([]byte(nil), h.c...),
		final:   h.final,
		leaf:    h.leaf,
		pos:     h.pos,
		leafLen: h.leafLen,
		state:   h.state,
	}
}

// Reset reinitializes h for reuse with the same customization string passed to
// [New]. Reset does not guarantee erasure of the previous hashing state.
func (h *Hasher) Reset() {
	h.final = sponge{}
	h.leaf = sponge{}
	h.pos = 0
	h.leafLen = 0
	h.state = stateSingle
}

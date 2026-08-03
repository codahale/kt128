package kt128

import "hash"

// Clone returns an independent copy of h at its current absorption or squeeze
// position, including a copy of its customization string. The returned error is
// always nil.
func (h *Hasher) Clone() (hash.Cloner, error) {
	return &Hasher{
		c:       append([]byte(nil), h.c...),
		final:   h.final,
		leaf:    h.leaf,
		pos:     h.pos,
		leafLen: h.leafLen,
		state:   h.state,
	}, nil
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

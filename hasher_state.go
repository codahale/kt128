package kt128

// Clone returns an independent copy of h at its current absorption or squeeze
// position, including a copy of its customization string. Clone panics if h has
// been cleared.
func (h *Hasher) Clone() *Hasher {
	h.checkNotCleared()
	return &Hasher{
		c:       append([]byte(nil), h.c...),
		final:   h.final,
		leaf:    h.leaf,
		pos:     h.pos,
		leafLen: h.leafLen,
		state:   h.state,
	}
}

// Reset makes a best effort to zero h's message-dependent state and resets it
// for reuse with the same customization string passed to [New]. Reset panics
// if h has been cleared.
func (h *Hasher) Reset() {
	h.checkNotCleared()
	h.final.wipe()
	h.leaf.wipe()
	h.pos = 0
	h.leafLen = 0
	h.state = stateSingle
}

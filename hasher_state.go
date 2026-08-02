package kt128

// Clone returns an independent copy of h at its current absorption or squeeze
// position. The clone shares h's immutable CustomizationString. All other state
// is independent.
func (h *Hasher) Clone() *Hasher {
	return &Hasher{
		c:       h.c,
		final:   h.final,
		leaf:    h.leaf,
		pos:     h.pos,
		leafLen: h.leafLen,
		state:   h.state,
	}
}

// Reset makes a best effort to zero h's message-dependent state and resets it
// for reuse with the same CustomizationString passed to [New]. If that
// CustomizationString has been cleared, the next Read will panic.
func (h *Hasher) Reset() {
	h.final.wipe()
	h.leaf.wipe()
	h.pos = 0
	h.leafLen = 0
	h.state = stateSingle
}

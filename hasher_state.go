package kt128

// Clone returns an independent copy of h at its current absorption or squeeze
// position. The clone shares h's caller-owned customization slice; the caller
// must not modify that slice while either Hasher may be used. All other state is
// independent.
func (h *Hasher) Clone() *Hasher {
	return &Hasher{
		c:         h.c,
		final:     h.final,
		leaf:      h.leaf,
		pos:       h.pos,
		leafCount: h.leafCount,
		leafLen:   h.leafLen,
		ds:        h.ds,
		state:     h.state,
	}
}

// Reset makes a best effort to zero h's message-dependent state and resets it
// for reuse with the same caller-owned customization string passed to [New].
func (h *Hasher) Reset() {
	h.final.wipe()
	h.leaf.wipe()
	h.pos = 0
	h.ds = 0
	h.leafCount = 0
	h.leafLen = 0
	h.state = stateSingle
}

package kt128

import (
	"crypto/subtle"
	"slices"
)

func (h *Hasher) Clone() *Hasher {
	return &Hasher{
		buf:        slices.Clone(h.buf),
		c:          h.c,
		final:      h.final,
		pending:    h.pending,
		pos:        h.pos,
		leafCount:  h.leafCount,
		pendingLen: h.pendingLen,
		ds:         h.ds,
		state:      h.state,
	}
}

// Reset resets the Hasher to its initial state, preserving the customization
// string passed to New. Like the standard library hash implementations, it
// does not scrub buffered message data from memory.
func (h *Hasher) Reset() {
	h.buf = h.buf[:0]
	h.final.reset()
	h.pos = 0
	h.ds = 0
	h.leafCount = 0
	h.pendingLen = 0 // pending's contents are fully overwritten before reuse
	h.state = stateSingle
}

// Clear zeros all message-derived state owned by the Hasher and resets it for
// reuse, preserving the customization string passed to New. Unlike Reset, it
// scrubs the full backing array of the buffered message before releasing it.
func (h *Hasher) Clear() {
	c := h.c
	if cap(h.buf) > 0 {
		clear(h.buf[:cap(h.buf)])
	}
	*h = Hasher{c: c}
}

// Equal returns 1 if h and other represent identical states, 0 otherwise.
func (h *Hasher) Equal(other *Hasher) int {
	var a, b [32]byte
	_, _ = h.Clone().Read(a[:])
	_, _ = other.Clone().Read(b[:])
	return subtle.ConstantTimeCompare(a[:], b[:])
}

// customSuffix appends C || length_encode(|C|) to dst and returns the result.

package kt128

import (
	"crypto/subtle"
	"slices"
)

// Clone returns an independent copy of h at its current absorption or squeeze
// position. Mutating either Hasher afterward does not affect the other.
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

// Reset resets h to its initial state while preserving the customization string
// passed to [New]. Like the standard library hash implementations, Reset does
// not scrub buffered message data from memory; use [Hasher.Clear] when that is
// required.
func (h *Hasher) Reset() {
	h.buf = h.buf[:0]
	h.final.reset()
	h.pos = 0
	h.ds = 0
	h.leafCount = 0
	h.pendingLen = 0 // pending's contents are fully overwritten before reuse
	h.state = stateSingle
}

// Clear zeros all message-derived state owned by h and resets it for reuse while
// preserving the customization string passed to [New]. Unlike [Hasher.Reset],
// Clear scrubs the full backing array of the buffered message before releasing
// it.
func (h *Hasher) Clear() {
	c := h.c
	if cap(h.buf) > 0 {
		clear(h.buf[:cap(h.buf)])
	}
	*h = Hasher{c: c}
}

// Equal returns 1 if the next 32 output bytes from h and other are equal, and 0
// otherwise. It does not modify either Hasher. The comparison is constant-time
// with respect to the contents of the inputs absorbed by the two hashers.
func (h *Hasher) Equal(other *Hasher) int {
	var a, b [32]byte
	_, _ = h.Clone().Read(a[:])
	_, _ = other.Clone().Read(b[:])
	return subtle.ConstantTimeCompare(a[:], b[:])
}

// customSuffix appends C || length_encode(|C|) to dst and returns the result.

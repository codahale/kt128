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
		c:          slices.Clone(h.c),
		final:      h.final,
		pending:    h.pending,
		pos:        h.pos,
		leafCount:  h.leafCount,
		pendingLen: h.pendingLen,
		ds:         h.ds,
		state:      h.state,
	}
}

// Reset resets h for reuse while preserving the customization string passed to
// [New]. Like the standard library hash implementations, Reset does not scrub
// buffered message data from memory. Call [Hasher.Clear] after the application
// is finished with h when its state may contain sensitive data.
func (h *Hasher) Reset() {
	h.buf = h.buf[:0]
	h.final.reset()
	h.pos = 0
	h.ds = 0
	h.leafCount = 0
	h.pendingLen = 0 // pending's contents are fully overwritten before reuse
	h.state = stateSingle
}

// Clear makes a best effort to zero all state owned by h, including the
// customization string. It is intended as final cleanup after the application
// is finished with h, not as a substitute for [Hasher.Reset]. After calling
// Clear, discard h and create a new Hasher for subsequent hashing.
//
// Clear cannot erase caller-owned input or output buffers, independent clones,
// copies made by the Go compiler or runtime, or values left in registers. Each
// clone must be cleared separately.
//
//go:noinline
func (h *Hasher) Clear() {
	if cap(h.buf) > 0 {
		wipeBytes(h.buf[:cap(h.buf)])
	}
	if cap(h.c) > 0 {
		wipeBytes(h.c[:cap(h.c)])
	}
	*h = Hasher{}
}

// Equal returns 1 if the next 32 output bytes from h and other are equal, and 0
// otherwise. It does not modify either Hasher. For fixed input lengths and
// lifecycle states, its work is independent of the absorbed byte values. It
// does not conceal input lengths, buffered lengths, or lifecycle states.
func (h *Hasher) Equal(other *Hasher) int {
	aClone, bClone := h.Clone(), other.Clone()
	defer aClone.Clear()
	defer bClone.Clear()

	var a, b [32]byte
	defer wipeBytes(a[:])
	defer wipeBytes(b[:])
	_, _ = aClone.Read(a[:])
	_, _ = bClone.Read(b[:])
	return subtle.ConstantTimeCompare(a[:], b[:])
}

// customSuffix appends C || length_encode(|C|) to dst and returns the result.

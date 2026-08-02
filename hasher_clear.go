package kt128

// Clear makes a best effort to overwrite h's customization string and
// message-dependent state, then permanently invalidates h. All subsequent
// exported operations on h except Clear and BlockSize panic. Clear is
// idempotent and may be called on a nil Hasher.
//
// Clearing a Hasher does not affect Hashers created by [Hasher.Clone].
//
// As with any best-effort clearing operation in Go, Clear cannot erase copies
// made by the compiler or runtime, the caller's original customization slice,
// or values left in registers.
func (h *Hasher) Clear() {
	if h == nil || h.state == stateCleared {
		return
	}
	h.final.wipe()
	h.leaf.wipe()
	wipeBytes(h.c)
	h.c = nil
	h.pos = 0
	h.leafLen = 0
	h.state = stateCleared
}

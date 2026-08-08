package kt128

func (h *state) cloneInto(dst *state) {
	h.cloneIntoBorrowingCustomization(dst)
	dst.c = append([]byte(nil), h.c...)
}

// cloneIntoBorrowingCustomization copies h into dst while sharing h's
// immutable customization string. It is only for synchronous operations that
// do not let dst escape.
func (h *state) cloneIntoBorrowingCustomization(dst *state) {
	dst.c = h.c
	dst.final = h.final
	dst.leaf = h.leaf
	dst.pos = h.pos
	dst.leafLen = h.leafLen
	dst.phase = h.phase
}

func (h *state) reset() {
	h.final = sponge{}
	h.leaf = sponge{}
	h.pos = 0
	h.leafLen = 0
	h.phase = stateSingle
}

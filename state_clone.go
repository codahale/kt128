package kt128

func (h *state) cloneInto(dst *state) {
	dst.c = append([]byte(nil), h.c...)
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

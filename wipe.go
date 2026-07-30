package kt128

import "runtime"

// wipeBytes is kept out of line so callers cannot prove that b is dead and
// eliminate the stores. Go and the runtime may still make copies outside the
// package's control, so this remains a best-effort operation.
//
//go:noinline
func wipeBytes(b []byte) {
	clear(b)
	runtime.KeepAlive(b)
}

// wipe clears a sponge held in addressable memory.
//
//go:noinline
func (s *sponge) wipe() {
	clear(s.a[:])
	s.pos = 0
	runtime.KeepAlive(s)
}

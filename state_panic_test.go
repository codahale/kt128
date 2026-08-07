package kt128

import "testing"

// mustPanic runs fn and fails unless it panics with exactly want.
func mustPanic(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		switch r := recover(); {
		case r == nil:
			t.Fatalf("expected panic %q, got none", want)
		case r != want:
			t.Fatalf("expected panic %q, got %v", want, r)
		}
	}()
	fn()
}

// TestPanics covers the package's reachable panic contracts. The "invalid final
// tail length" panic in absorbAll is an unreachable defensive assertion:
// fastLoopAbsorb168 always consumes a whole number of rate-sized blocks, so the
// remaining tail is always shorter than the rate. There is no input that
// triggers it without a bug in fastLoopAbsorb168 itself, so it is not exercised
// here.
func TestPanics(t *testing.T) {
	// Writing after finalization (the first Read) is forbidden.
	t.Run("write after finalize", func(t *testing.T) {
		h := NewXOF(nil)
		if _, err := h.Read(make([]byte, 32)); err != nil {
			t.Fatalf("Read: %v", err)
		}
		mustPanic(t, "kt128: XOF is finalized", func() {
			_, _ = h.Write([]byte("x"))
		})
	})

	// The message position must remain exact because finalization derives the
	// tree leaf count from it. Reject the write before absorbing any bytes.
	t.Run("message length overflow", func(t *testing.T) {
		h := NewXOF(nil)
		h.phase = stateTree
		h.pos = ^uint64(0)
		h.final.a[0] = 1
		h.leaf.a[0] = 2
		h.leafLen = 37

		mustPanic(t, "kt128: message length exceeds 2^64-1 bytes", func() {
			_, _ = h.Write([]byte{0xA5})
		})

		if h.pos != ^uint64(0) || h.final.a[0] != 1 || h.leaf.a[0] != 2 || h.leafLen != 37 {
			t.Fatalf("overflowing Write modified the XOF: %#v", h)
		}
	})

	// Chain values must be absorbed at a lane-aligned (multiple-of-8) position.
	t.Run("absorbCV on non-lane-aligned state", func(t *testing.T) {
		var s, src sponge
		s.pos = 1
		mustPanic(t, "kt128: absorbCV on non-lane-aligned state", func() {
			s.absorbCV(&src)
		})
	})

	t.Run("absorbCVs on non-lane-aligned state", func(t *testing.T) {
		var s sponge
		s.pos = 1
		mustPanic(t, "kt128: absorbCVs on non-lane-aligned state", func() {
			s.absorbCVs(make([]byte, 32))
		})
	})

	t.Run("absorbCVs with partial CV", func(t *testing.T) {
		var s sponge
		mustPanic(t, "kt128: absorbCVs input length is not a multiple of 32", func() {
			s.absorbCVs(make([]byte, 33))
		})
	})
}

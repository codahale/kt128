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

// TestPanics covers the package's documented panic contracts. The fifth panic,
// "invalid final tail length" in absorbAll, is an unreachable defensive
// assertion: fastLoopAbsorb168 always consumes a whole number of rate-sized
// blocks, so the remaining tail is always shorter than the rate. There is no
// input that triggers it without a bug in fastLoopAbsorb168 itself, so it is not
// exercised here.
func TestPanics(t *testing.T) {
	// Writing after finalization (the first Read) is forbidden.
	t.Run("write after finalize", func(t *testing.T) {
		h := New()
		if _, err := h.Read(make([]byte, 32)); err != nil {
			t.Fatalf("Read: %v", err)
		}
		mustPanic(t, "kt128: Hasher is finalized", func() {
			_, _ = h.Write([]byte("x"))
		})
	})

	// Setting the customization string after finalization is forbidden.
	t.Run("set customization after finalize", func(t *testing.T) {
		h := New()
		if _, err := h.Read(make([]byte, 32)); err != nil {
			t.Fatalf("Read: %v", err)
		}
		mustPanic(t, "kt128: Hasher is finalized", func() {
			h.SetCustomizationString([]byte("x"))
		})
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
}

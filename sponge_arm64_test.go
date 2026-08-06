//go:build arm64 && !purego

package kt128

import (
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// TestFastLoopAbsorbRejectsRaggedLength verifies the wrapper's guard on the
// NEON absorb loop, whose exact-count termination would run a ragged or zero
// length off the end of the input.
func TestFastLoopAbsorbRejectsRaggedLength(t *testing.T) {
	if !cpuid.HasSHA3 {
		t.Skip("no SHA3 extension")
	}
	for _, n := range []int{0, 1, rate - 1, rate + 1, 2*rate + 7} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("len %d: expected panic, got none", n)
				}
			}()
			var s sponge
			fastLoopAbsorb168x1Arch(&s, make([]byte, n))
		}()
	}
}

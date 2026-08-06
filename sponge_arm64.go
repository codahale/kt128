//go:build arm64 && !purego

package kt128

import (
	"unsafe"

	"github.com/codahale/kt128/internal/cpuid"
)

//go:noescape
func p1600(a *sponge)

func permute12x1Arch(s *sponge) bool {
	if !cpuid.HasSHA3 {
		return false
	}
	p1600(s)
	return true
}

//go:noescape
func fastLoopAbsorb168x1(s *sponge, in *byte, n int)

func fastLoopAbsorb168x1Arch(s *sponge, in []byte) bool {
	if !cpuid.HasSHA3 {
		return false
	}
	// The NEON loop terminates only on an exact byte-count match, so a ragged
	// or zero length would run it off the end of the input; the amd64 loop
	// bound-checks each stripe and tolerates both.
	if len(in) == 0 || len(in)%rate != 0 {
		panic("kt128: fastLoopAbsorb168 requires a positive whole number of rate blocks")
	}
	fastLoopAbsorb168x1(s, unsafe.SliceData(in), len(in))
	return true
}

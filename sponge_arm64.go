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
	fastLoopAbsorb168x1(s, unsafe.SliceData(in), len(in))
	return true
}

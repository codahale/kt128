//go:build amd64 && !purego && (linux || darwin)

package kt128

import "github.com/codahale/kt128/internal/cpuid"

func assemblySafetyPermuters() []assemblySafetyPermuter {
	var out []assemblySafetyPermuter
	if cpuid.HasBMI2 {
		out = append(out, assemblySafetyPermuter{"p1600", p1600})
	}
	if cpuid.HasAVX512 {
		out = append(out, assemblySafetyPermuter{"p1600AVX512", p1600AVX512})
	}
	return out
}

func assemblySafetyAbsorbers() []assemblySafetyAbsorber {
	var out []assemblySafetyAbsorber
	if cpuid.HasBMI2 {
		out = append(out, assemblySafetyAbsorber{"fastLoopAbsorb168x1", fastLoopAbsorb168x1})
	}
	if cpuid.HasAVX512 {
		out = append(out, assemblySafetyAbsorber{"fastLoopAbsorb168x1AVX512", fastLoopAbsorb168x1AVX512})
	}
	return out
}

//go:build arm64 && !purego && (linux || darwin)

package kt128

import "github.com/codahale/kt128/internal/cpuid"

func assemblySafetyPermuters() []assemblySafetyPermuter {
	if !cpuid.HasSHA3 {
		return nil
	}
	return []assemblySafetyPermuter{{"p1600", p1600}}
}

func assemblySafetyAbsorbers() []assemblySafetyAbsorber {
	if !cpuid.HasSHA3 {
		return nil
	}
	return []assemblySafetyAbsorber{{"fastLoopAbsorb168x1", fastLoopAbsorb168x1}}
}

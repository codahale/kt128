//go:build amd64 && !purego

package cpuid

func hasAVX2() bool
func hasAVX512VL() bool

var HasAVX2 = hasAVX2()

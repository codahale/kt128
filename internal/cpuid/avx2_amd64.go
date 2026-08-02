//go:build amd64 && !purego

package cpuid

import "golang.org/x/sys/cpu"

var HasAVX2 = cpu.X86.HasAVX2

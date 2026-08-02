//go:build amd64 && !purego

package cpuid

import "golang.org/x/sys/cpu"

var HasAVX512 = cpu.X86.HasAVX512 && cpu.X86.HasAVX512F && cpu.X86.HasAVX512VL

//go:build amd64 && !purego

package cpuid

import "golang.org/x/sys/cpu"

var HasBMI2 = cpu.X86.HasBMI2

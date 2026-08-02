//go:build !purego

package cpuid

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

var (
	HasAVX2   = runtime.GOARCH == "amd64" && cpu.X86.HasAVX2
	HasAVX512 = runtime.GOARCH == "amd64" && cpu.X86.HasAVX512 && cpu.X86.HasAVX512F && cpu.X86.HasAVX512VL
	HasBMI2   = runtime.GOARCH == "amd64" && cpu.X86.HasBMI2
	HasSHA3   = runtime.GOARCH == "arm64" && cpu.ARM64.HasSHA3
)

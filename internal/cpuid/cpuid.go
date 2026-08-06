//go:build !purego

package cpuid

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

var (
	HasAVX2 = runtime.GOARCH == "amd64" && cpu.X86.HasAVX2
	// The leaf kernels' KMOVB is AVX-512DQ, so the gate requires it alongside
	// F and VL; every CPU shipping VL also ships DQ, but the gate should match
	// the instructions actually used.
	HasAVX512 = runtime.GOARCH == "amd64" && cpu.X86.HasAVX512 && cpu.X86.HasAVX512F && cpu.X86.HasAVX512VL && cpu.X86.HasAVX512DQ
	HasBMI2   = runtime.GOARCH == "amd64" && cpu.X86.HasBMI2
	HasSHA3   = runtime.GOARCH == "arm64" && cpu.ARM64.HasSHA3
)

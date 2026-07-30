//go:build arm64 && !purego

package cpuid

import "golang.org/x/sys/cpu"

var HasSHA3 = cpu.ARM64.HasSHA3

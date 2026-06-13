//go:build amd64 && !purego

package kt128

import (
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// These mirror FuzzP1600 / FuzzFastLoopAbsorb168x1 for the AVX-512 kernels,
// which are a distinct code path selected at runtime. They are skipped on hosts
// without AVX-512 (including under Rosetta), so they only provide coverage on an
// AVX-512-capable runner.

// FuzzP1600AVX512 fuzzes the AVX-512 scalar permutation against the pure-Go
// reference.
func FuzzP1600AVX512(f *testing.F) {
	if !cpuid.HasAVX512 {
		f.Skip("no AVX-512 on this host")
	}
	addPermuteSeeds(f)
	f.Fuzz(func(t *testing.T, data []byte) {
		checkPermute(t, p1600AVX512, data)
	})
}

// FuzzFastLoopAbsorb168x1AVX512 fuzzes the AVX-512 absorb-permute loop against
// the pure-Go reference.
func FuzzFastLoopAbsorb168x1AVX512(f *testing.F) {
	if !cpuid.HasAVX512 {
		f.Skip("no AVX-512 on this host")
	}
	f.Add(make([]byte, stateBytes), make([]byte, rate))
	f.Add(make([]byte, stateBytes), make([]byte, 48*rate))
	f.Add(splitmixBytes(stateBytes, 3), splitmixBytes(7*rate, 4))
	f.Fuzz(func(t *testing.T, stateData, inData []byte) {
		checkAbsorb(t, fastLoopAbsorb168x1AVX512, stateData, inData)
	})
}

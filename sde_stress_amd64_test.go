//go:build amd64 && !purego

package kt128

import (
	"os"
	"strconv"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// TestSDEStress runs fuzz-equivalent differential checks in one process. Intel
// SDE's Pin runtime is not reliable around the subprocesses created by Go's
// fuzz coordinator, so the nightly SDE jobs use this deterministic driver and
// reserve coverage-guided fuzzing for native runners.
func TestSDEStress(t *testing.T) {
	target := os.Getenv("KT128_STRESS_TARGET")
	if target == "" {
		t.Skip("KT128_STRESS_TARGET is not set")
	}
	iterations, err := strconv.Atoi(os.Getenv("KT128_STRESS_ITERATIONS"))
	if err != nil || iterations < 1 {
		t.Fatalf("invalid KT128_STRESS_ITERATIONS %q", os.Getenv("KT128_STRESS_ITERATIONS"))
	}
	seed, err := strconv.ParseUint(os.Getenv("KT128_STRESS_SEED"), 10, 64)
	if err != nil {
		t.Fatalf("invalid KT128_STRESS_SEED %q: %v", os.Getenv("KT128_STRESS_SEED"), err)
	}

	t.Logf("target=%s seed=%d iterations=%d", target, seed, iterations)
	for i := range iterations {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			r := newSDEStressRNG(seed, i)
			r.run(t, target)
		})
	}
}

type sdeStressRNG uint64

func newSDEStressRNG(seed uint64, iteration int) sdeStressRNG {
	return sdeStressRNG(sdeStressMix(seed + uint64(iteration)*0x9E3779B97F4A7C15))
}

func (r *sdeStressRNG) next() uint64 {
	*r += 0x9E3779B97F4A7C15
	return sdeStressMix(uint64(*r))
}

func sdeStressMix(z uint64) uint64 {
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func (r *sdeStressRNG) bytes(n int) []byte {
	return splitmixBytes(n, r.next())
}

// length returns a dense small length seven times out of eight and otherwise
// selects an exact state-machine boundary.
func (r *sdeStressRNG) length(smallMax int, boundaries []int) int {
	if r.next()&7 != 0 {
		return int(r.next() % uint64(smallMax+1))
	}
	return boundaries[int(r.next()%uint64(len(boundaries)))]
}

func (r *sdeStressRNG) run(t *testing.T, target string) {
	t.Helper()
	switch target {
	case "FuzzXOF":
		msgLen := r.length(1024, []int{
			0, 1, rate - 1, rate, rate + 1,
			ChunkSize - 1, ChunkSize, ChunkSize + 1,
			2 * ChunkSize, 3 * ChunkSize, 7 * ChunkSize,
			8 * ChunkSize, 9 * ChunkSize, 11*ChunkSize + 123, 96 * 1024,
		})
		customLen := r.length(256, []int{
			0, 1, rate - 1, rate, rate + 1,
			ChunkSize - 1, ChunkSize, ChunkSize + 1, 2*ChunkSize + 3,
		})
		checkXOFInput(t, r.bytes(msgLen), r.bytes(customLen), uint16(r.next()), uint16(r.next()))

	case "FuzzXOFLifecycle":
		custom0 := r.bytes(r.length(128, []int{0, 1, rate, ChunkSize, lcMaxMsg}))
		custom1 := r.bytes(r.length(128, []int{0, 1, rate, ChunkSize, lcMaxMsg}))
		program := r.bytes(r.length(1024, []int{0, 1, 4, 64, 256, 512, 1024}))
		executeLifecycle(t, [][]byte{custom0, custom1}, decodeOps(program))

	case "FuzzP1600AVX512":
		if !cpuid.HasAVX512 {
			t.Fatal("AVX-512 is unavailable")
		}
		data := r.bytes(r.length(2*stateBytes, []int{0, 1, stateBytes - 1, stateBytes, stateBytes + 1, 2 * stateBytes}))
		checkPermute(t, p1600AVX512, data)

	case "FuzzFastLoopAbsorb168x1AVX512":
		if !cpuid.HasAVX512 {
			t.Fatal("AVX-512 is unavailable")
		}
		stateData := r.bytes(r.length(2*stateBytes, []int{0, 1, stateBytes - 1, stateBytes, stateBytes + 1, 2 * stateBytes}))
		inData := r.bytes(r.length(4*rate, []int{
			0, 1, rate - 1, rate, rate + 1,
			7 * rate, 48 * rate, maxFuzzStripes * rate, maxFuzzStripes*rate + rate - 1,
		}))
		checkAbsorb(t, fastLoopAbsorb168x1AVX512, stateData, inData)

	default:
		t.Fatalf("unknown KT128_STRESS_TARGET %q", target)
	}
}

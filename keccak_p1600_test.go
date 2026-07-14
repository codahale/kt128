package kt128

import (
	"math/bits"
	"testing"
)

// keccakP1600x12Reference is the straightforward, table-driven Keccak-p[1600,12]
// used as a differential oracle for the optimized keccakP1600x12. It is a direct
// transcription of the algorithm and is intentionally simple, not fast.
func keccakP1600x12Reference(a *[lanes]uint64) {
	rho := [24]uint{
		1, 3, 6, 10, 15, 21,
		28, 36, 45, 55, 2, 14,
		27, 41, 56, 8, 25, 43,
		62, 18, 39, 61, 20, 44,
	}
	pi := [24]int{
		10, 7, 11, 17, 18, 3,
		5, 16, 8, 21, 24, 4,
		15, 23, 19, 13, 12, 2,
		20, 14, 22, 9, 6, 1,
	}

	var c [5]uint64
	for round := 12; round < 24; round++ {
		for i := range 5 {
			c[i] = a[i] ^ a[i+5] ^ a[i+10] ^ a[i+15] ^ a[i+20]
		}
		for i := range 5 {
			t := c[(i+4)%5] ^ bits.RotateLeft64(c[(i+1)%5], 1)
			for j := 0; j < lanes; j += 5 {
				a[j+i] ^= t
			}
		}
		t := a[1]
		for i := range 24 {
			j := pi[i]
			c0 := a[j]
			a[j] = bits.RotateLeft64(t, int(rho[i]))
			t = c0
		}
		for j := 0; j < lanes; j += 5 {
			for i := range 5 {
				c[i] = a[j+i]
			}
			for i := range 5 {
				a[j+i] = c[i] ^ (^c[(i+1)%5] & c[(i+2)%5])
			}
		}
		a[0] ^= roundConstants[round]
	}
}

// TestKeccakP1600x12 checks the optimized permutation against the reference over
// a range of states, including all-zero, all-one, single-bit, and pseudorandom.
func TestKeccakP1600x12(t *testing.T) {
	states := make([][lanes]uint64, 0, 64)

	// All zero and all ones.
	states = append(states, [lanes]uint64{})
	var ones [lanes]uint64
	for i := range ones {
		ones[i] = ^uint64(0)
	}
	states = append(states, ones)

	// Single set bit in each lane.
	for lane := range lanes {
		var s [lanes]uint64
		s[lane] = 1
		states = append(states, s)
		s[lane] = 1 << 63
		states = append(states, s)
	}

	// Pseudorandom states via a splitmix64 stream.
	var seed uint64 = 0x9E3779B97F4A7C15
	next := func() uint64 {
		seed += 0x9E3779B97F4A7C15
		z := seed
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		return z ^ (z >> 31)
	}
	for range 32 {
		var s [lanes]uint64
		for i := range s {
			s[i] = next()
		}
		states = append(states, s)
	}

	for idx, s := range states {
		got := s
		want := s
		keccakP1600x12(&got)
		keccakP1600x12Reference(&want)
		if got != want {
			t.Fatalf("state %d: permutation mismatch\n got  %016x\n want %016x", idx, got, want)
		}
	}
}

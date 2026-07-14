//go:build (amd64 || arm64) && !purego

package kt128

import (
	"encoding/binary"
	"testing"
)

// These fuzz targets pit the hand-written assembly single-lane primitives
// directly against keccakP1600x12, the pure-Go permutation. That reference is
// itself validated against an independent table-driven transcription in
// TestKeccakP1600x12, so a divergence here localizes a fault to the assembly
// rather than only surfacing it transitively through the RFC vectors.

// posSentinel is written to sponge.pos before each call so a kernel that writes
// past its 25 lanes (200 bytes) into the adjacent pos field is caught.
const posSentinel = 0x0BAD

// maxFuzzStripes caps generated absorb inputs. The real leaf path absorbs 48
// stripes, so this comfortably covers the multi-stripe state evolution while
// keeping iterations fast.
const maxFuzzStripes = 64

// stateFromBytes packs up to 200 bytes of fuzz input into a 25-lane Keccak
// state, zero-padding short inputs and ignoring any excess.
func stateFromBytes(data []byte) [lanes]uint64 {
	var b [stateBytes]byte
	copy(b[:], data)
	var a [lanes]uint64
	for i := range a {
		a[i] = binary.LittleEndian.Uint64(b[i*8:])
	}
	return a
}

// referenceAbsorb168 XORs each full 168-byte stripe of in into the rate lanes
// and applies the pure-Go permutation, mirroring fastLoopAbsorb168x1.
func referenceAbsorb168(a *[lanes]uint64, in []byte) {
	for off := 0; off+rate <= len(in); off += rate {
		for lane := range rate >> 3 {
			a[lane] ^= binary.LittleEndian.Uint64(in[off+lane*8:])
		}
		keccakP1600x12(a)
	}
}

// splitmixBytes returns n deterministic pseudorandom bytes derived from seed,
// for use as a fuzz corpus seed (Math.random is unavailable and undesirable here).
func splitmixBytes(n int, seed uint64) []byte {
	b := make([]byte, n)
	x := seed
	for i := range b {
		x += 0x9E3779B97F4A7C15
		z := x
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		z ^= z >> 31
		b[i] = byte(z)
	}
	return b
}

// checkPermute runs an assembly single-lane permutation against the pure-Go
// reference for one fuzz input and reports any divergence.
func checkPermute(t *testing.T, permute func(*sponge), data []byte) {
	t.Helper()

	in := stateFromBytes(data)
	var got sponge
	got.a = in
	got.pos = posSentinel
	want := in

	permute(&got)
	keccakP1600x12(&want)

	if got.a != want {
		t.Fatalf("permutation mismatch\n in   %016x\n got  %016x\n want %016x", in, got.a, want)
	}
	if got.pos != posSentinel {
		t.Fatalf("permutation modified pos: got %#x, want %#x", got.pos, posSentinel)
	}
}

// checkAbsorb runs an assembly fused absorb-permute loop against the pure-Go
// reference for one fuzz input. stateData seeds the initial state; inData is
// truncated to a positive whole number of 168-byte stripes, matching the
// kernels' precondition (arm64's loop is do-while, so n must be >= 168).
func checkAbsorb(t *testing.T, absorb func(*sponge, *byte, int), stateData, inData []byte) {
	t.Helper()

	nStripes := min(max(len(inData)/rate, 1), maxFuzzStripes)
	n := nStripes * rate
	in := make([]byte, n)
	copy(in, inData)

	var got sponge
	got.a = stateFromBytes(stateData)
	got.pos = posSentinel
	absorb(&got, &in[0], n)

	want := stateFromBytes(stateData)
	referenceAbsorb168(&want, in)

	if got.a != want {
		t.Fatalf("absorb mismatch (%d stripes)\n got  %016x\n want %016x", nStripes, got.a, want)
	}
	if got.pos != posSentinel {
		t.Fatalf("absorb modified pos: got %#x, want %#x", got.pos, posSentinel)
	}
}

// addPermuteSeeds seeds a permutation fuzz corpus with the same edge states as
// TestKeccakP1600x12: all-zero, all-ones, and pseudorandom.
func addPermuteSeeds(f *testing.F) {
	f.Add(make([]byte, stateBytes))

	ones := make([]byte, stateBytes)
	for i := range ones {
		ones[i] = 0xFF
	}
	f.Add(ones)

	f.Add(splitmixBytes(stateBytes, 1))
	f.Add(splitmixBytes(stateBytes, 2))
}

// FuzzP1600 fuzzes the scalar assembly Keccak-p[1600,12] against the pure-Go
// reference.
func FuzzP1600(f *testing.F) {
	addPermuteSeeds(f)
	f.Fuzz(func(t *testing.T, data []byte) {
		checkPermute(t, p1600, data)
	})
}

// FuzzFastLoopAbsorb168x1 fuzzes the scalar assembly absorb-permute loop against
// the pure-Go reference, over arbitrary initial states and stripe counts.
func FuzzFastLoopAbsorb168x1(f *testing.F) {
	f.Add(make([]byte, stateBytes), make([]byte, rate))
	f.Add(make([]byte, stateBytes), make([]byte, 48*rate))
	f.Add(splitmixBytes(stateBytes, 3), splitmixBytes(7*rate, 4))
	f.Fuzz(func(t *testing.T, stateData, inData []byte) {
		checkAbsorb(t, fastLoopAbsorb168x1, stateData, inData)
	})
}

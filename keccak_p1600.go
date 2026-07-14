package kt128

import "math/bits"

const (
	stateBytes = 200
	lanes      = stateBytes / 8
	rate       = 168
)

func loadPartialLE(in []byte) uint64 {
	var v uint64
	for i := range in {
		v |= uint64(in[i]) << (8 * i)
	}
	return v
}

func xorByteInWord(w *uint64, pos int, b byte) {
	shift := uint((pos & 7) << 3)
	*w ^= uint64(b) << shift
}

var roundConstants = [24]uint64{
	0x0000000000000001,
	0x0000000000008082,
	0x800000000000808A,
	0x8000000080008000,
	0x000000000000808B,
	0x0000000080000001,
	0x8000000080008081,
	0x8000000000008009,
	0x000000000000008A,
	0x0000000000000088,
	0x0000000080008009,
	0x000000008000000A,
	0x000000008000808B,
	0x800000000000008B,
	0x8000000000008089,
	0x8000000000008003,
	0x8000000000008002,
	0x8000000000000080,
	0x000000000000800A,
	0x800000008000000A,
	0x8000000080008081,
	0x8000000000008080,
	0x0000000080000001,
	0x8000000080008008,
}

// keccakP1600x12 applies Keccak-p[1600, 12]: the last 12 rounds of Keccak-f.
//
// The five steps of each round are fused into two and unrolled four rounds at a
// time, holding the whole 25-lane state in locals so it stays in registers and
// every ρ rotation is a constant. This is the standard unrolled translation of
// the public-domain Keccak reference (Keccak-inplace.c); the only change from a
// full Keccak-f is the loop starting at round 12.
func keccakP1600x12(a *[lanes]uint64) {
	var t, bc0, bc1, bc2, bc3, bc4, d0, d1, d2, d3, d4 uint64

	for i := 12; i < 24; i += 4 {
		// Round 1
		bc0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20]
		bc1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21]
		bc2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22]
		bc3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23]
		bc4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24]
		d0 = bc4 ^ bits.RotateLeft64(bc1, 1)
		d1 = bc0 ^ bits.RotateLeft64(bc2, 1)
		d2 = bc1 ^ bits.RotateLeft64(bc3, 1)
		d3 = bc2 ^ bits.RotateLeft64(bc4, 1)
		d4 = bc3 ^ bits.RotateLeft64(bc0, 1)

		bc0 = a[0] ^ d0
		t = a[6] ^ d1
		bc1 = bits.RotateLeft64(t, 44)
		t = a[12] ^ d2
		bc2 = bits.RotateLeft64(t, 43)
		t = a[18] ^ d3
		bc3 = bits.RotateLeft64(t, 21)
		t = a[24] ^ d4
		bc4 = bits.RotateLeft64(t, 14)
		a[0] = bc0 ^ (bc2 &^ bc1) ^ roundConstants[i]
		a[6] = bc1 ^ (bc3 &^ bc2)
		a[12] = bc2 ^ (bc4 &^ bc3)
		a[18] = bc3 ^ (bc0 &^ bc4)
		a[24] = bc4 ^ (bc1 &^ bc0)

		t = a[10] ^ d0
		bc2 = bits.RotateLeft64(t, 3)
		t = a[16] ^ d1
		bc3 = bits.RotateLeft64(t, 45)
		t = a[22] ^ d2
		bc4 = bits.RotateLeft64(t, 61)
		t = a[3] ^ d3
		bc0 = bits.RotateLeft64(t, 28)
		t = a[9] ^ d4
		bc1 = bits.RotateLeft64(t, 20)
		a[10] = bc0 ^ (bc2 &^ bc1)
		a[16] = bc1 ^ (bc3 &^ bc2)
		a[22] = bc2 ^ (bc4 &^ bc3)
		a[3] = bc3 ^ (bc0 &^ bc4)
		a[9] = bc4 ^ (bc1 &^ bc0)

		t = a[20] ^ d0
		bc4 = bits.RotateLeft64(t, 18)
		t = a[1] ^ d1
		bc0 = bits.RotateLeft64(t, 1)
		t = a[7] ^ d2
		bc1 = bits.RotateLeft64(t, 6)
		t = a[13] ^ d3
		bc2 = bits.RotateLeft64(t, 25)
		t = a[19] ^ d4
		bc3 = bits.RotateLeft64(t, 8)
		a[20] = bc0 ^ (bc2 &^ bc1)
		a[1] = bc1 ^ (bc3 &^ bc2)
		a[7] = bc2 ^ (bc4 &^ bc3)
		a[13] = bc3 ^ (bc0 &^ bc4)
		a[19] = bc4 ^ (bc1 &^ bc0)

		t = a[5] ^ d0
		bc1 = bits.RotateLeft64(t, 36)
		t = a[11] ^ d1
		bc2 = bits.RotateLeft64(t, 10)
		t = a[17] ^ d2
		bc3 = bits.RotateLeft64(t, 15)
		t = a[23] ^ d3
		bc4 = bits.RotateLeft64(t, 56)
		t = a[4] ^ d4
		bc0 = bits.RotateLeft64(t, 27)
		a[5] = bc0 ^ (bc2 &^ bc1)
		a[11] = bc1 ^ (bc3 &^ bc2)
		a[17] = bc2 ^ (bc4 &^ bc3)
		a[23] = bc3 ^ (bc0 &^ bc4)
		a[4] = bc4 ^ (bc1 &^ bc0)

		t = a[15] ^ d0
		bc3 = bits.RotateLeft64(t, 41)
		t = a[21] ^ d1
		bc4 = bits.RotateLeft64(t, 2)
		t = a[2] ^ d2
		bc0 = bits.RotateLeft64(t, 62)
		t = a[8] ^ d3
		bc1 = bits.RotateLeft64(t, 55)
		t = a[14] ^ d4
		bc2 = bits.RotateLeft64(t, 39)
		a[15] = bc0 ^ (bc2 &^ bc1)
		a[21] = bc1 ^ (bc3 &^ bc2)
		a[2] = bc2 ^ (bc4 &^ bc3)
		a[8] = bc3 ^ (bc0 &^ bc4)
		a[14] = bc4 ^ (bc1 &^ bc0)

		// Round 2
		bc0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20]
		bc1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21]
		bc2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22]
		bc3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23]
		bc4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24]
		d0 = bc4 ^ bits.RotateLeft64(bc1, 1)
		d1 = bc0 ^ bits.RotateLeft64(bc2, 1)
		d2 = bc1 ^ bits.RotateLeft64(bc3, 1)
		d3 = bc2 ^ bits.RotateLeft64(bc4, 1)
		d4 = bc3 ^ bits.RotateLeft64(bc0, 1)

		bc0 = a[0] ^ d0
		t = a[16] ^ d1
		bc1 = bits.RotateLeft64(t, 44)
		t = a[7] ^ d2
		bc2 = bits.RotateLeft64(t, 43)
		t = a[23] ^ d3
		bc3 = bits.RotateLeft64(t, 21)
		t = a[14] ^ d4
		bc4 = bits.RotateLeft64(t, 14)
		a[0] = bc0 ^ (bc2 &^ bc1) ^ roundConstants[i+1]
		a[16] = bc1 ^ (bc3 &^ bc2)
		a[7] = bc2 ^ (bc4 &^ bc3)
		a[23] = bc3 ^ (bc0 &^ bc4)
		a[14] = bc4 ^ (bc1 &^ bc0)

		t = a[20] ^ d0
		bc2 = bits.RotateLeft64(t, 3)
		t = a[11] ^ d1
		bc3 = bits.RotateLeft64(t, 45)
		t = a[2] ^ d2
		bc4 = bits.RotateLeft64(t, 61)
		t = a[18] ^ d3
		bc0 = bits.RotateLeft64(t, 28)
		t = a[9] ^ d4
		bc1 = bits.RotateLeft64(t, 20)
		a[20] = bc0 ^ (bc2 &^ bc1)
		a[11] = bc1 ^ (bc3 &^ bc2)
		a[2] = bc2 ^ (bc4 &^ bc3)
		a[18] = bc3 ^ (bc0 &^ bc4)
		a[9] = bc4 ^ (bc1 &^ bc0)

		t = a[15] ^ d0
		bc4 = bits.RotateLeft64(t, 18)
		t = a[6] ^ d1
		bc0 = bits.RotateLeft64(t, 1)
		t = a[22] ^ d2
		bc1 = bits.RotateLeft64(t, 6)
		t = a[13] ^ d3
		bc2 = bits.RotateLeft64(t, 25)
		t = a[4] ^ d4
		bc3 = bits.RotateLeft64(t, 8)
		a[15] = bc0 ^ (bc2 &^ bc1)
		a[6] = bc1 ^ (bc3 &^ bc2)
		a[22] = bc2 ^ (bc4 &^ bc3)
		a[13] = bc3 ^ (bc0 &^ bc4)
		a[4] = bc4 ^ (bc1 &^ bc0)

		t = a[10] ^ d0
		bc1 = bits.RotateLeft64(t, 36)
		t = a[1] ^ d1
		bc2 = bits.RotateLeft64(t, 10)
		t = a[17] ^ d2
		bc3 = bits.RotateLeft64(t, 15)
		t = a[8] ^ d3
		bc4 = bits.RotateLeft64(t, 56)
		t = a[24] ^ d4
		bc0 = bits.RotateLeft64(t, 27)
		a[10] = bc0 ^ (bc2 &^ bc1)
		a[1] = bc1 ^ (bc3 &^ bc2)
		a[17] = bc2 ^ (bc4 &^ bc3)
		a[8] = bc3 ^ (bc0 &^ bc4)
		a[24] = bc4 ^ (bc1 &^ bc0)

		t = a[5] ^ d0
		bc3 = bits.RotateLeft64(t, 41)
		t = a[21] ^ d1
		bc4 = bits.RotateLeft64(t, 2)
		t = a[12] ^ d2
		bc0 = bits.RotateLeft64(t, 62)
		t = a[3] ^ d3
		bc1 = bits.RotateLeft64(t, 55)
		t = a[19] ^ d4
		bc2 = bits.RotateLeft64(t, 39)
		a[5] = bc0 ^ (bc2 &^ bc1)
		a[21] = bc1 ^ (bc3 &^ bc2)
		a[12] = bc2 ^ (bc4 &^ bc3)
		a[3] = bc3 ^ (bc0 &^ bc4)
		a[19] = bc4 ^ (bc1 &^ bc0)

		// Round 3
		bc0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20]
		bc1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21]
		bc2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22]
		bc3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23]
		bc4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24]
		d0 = bc4 ^ bits.RotateLeft64(bc1, 1)
		d1 = bc0 ^ bits.RotateLeft64(bc2, 1)
		d2 = bc1 ^ bits.RotateLeft64(bc3, 1)
		d3 = bc2 ^ bits.RotateLeft64(bc4, 1)
		d4 = bc3 ^ bits.RotateLeft64(bc0, 1)

		bc0 = a[0] ^ d0
		t = a[11] ^ d1
		bc1 = bits.RotateLeft64(t, 44)
		t = a[22] ^ d2
		bc2 = bits.RotateLeft64(t, 43)
		t = a[8] ^ d3
		bc3 = bits.RotateLeft64(t, 21)
		t = a[19] ^ d4
		bc4 = bits.RotateLeft64(t, 14)
		a[0] = bc0 ^ (bc2 &^ bc1) ^ roundConstants[i+2]
		a[11] = bc1 ^ (bc3 &^ bc2)
		a[22] = bc2 ^ (bc4 &^ bc3)
		a[8] = bc3 ^ (bc0 &^ bc4)
		a[19] = bc4 ^ (bc1 &^ bc0)

		t = a[15] ^ d0
		bc2 = bits.RotateLeft64(t, 3)
		t = a[1] ^ d1
		bc3 = bits.RotateLeft64(t, 45)
		t = a[12] ^ d2
		bc4 = bits.RotateLeft64(t, 61)
		t = a[23] ^ d3
		bc0 = bits.RotateLeft64(t, 28)
		t = a[9] ^ d4
		bc1 = bits.RotateLeft64(t, 20)
		a[15] = bc0 ^ (bc2 &^ bc1)
		a[1] = bc1 ^ (bc3 &^ bc2)
		a[12] = bc2 ^ (bc4 &^ bc3)
		a[23] = bc3 ^ (bc0 &^ bc4)
		a[9] = bc4 ^ (bc1 &^ bc0)

		t = a[5] ^ d0
		bc4 = bits.RotateLeft64(t, 18)
		t = a[16] ^ d1
		bc0 = bits.RotateLeft64(t, 1)
		t = a[2] ^ d2
		bc1 = bits.RotateLeft64(t, 6)
		t = a[13] ^ d3
		bc2 = bits.RotateLeft64(t, 25)
		t = a[24] ^ d4
		bc3 = bits.RotateLeft64(t, 8)
		a[5] = bc0 ^ (bc2 &^ bc1)
		a[16] = bc1 ^ (bc3 &^ bc2)
		a[2] = bc2 ^ (bc4 &^ bc3)
		a[13] = bc3 ^ (bc0 &^ bc4)
		a[24] = bc4 ^ (bc1 &^ bc0)

		t = a[20] ^ d0
		bc1 = bits.RotateLeft64(t, 36)
		t = a[6] ^ d1
		bc2 = bits.RotateLeft64(t, 10)
		t = a[17] ^ d2
		bc3 = bits.RotateLeft64(t, 15)
		t = a[3] ^ d3
		bc4 = bits.RotateLeft64(t, 56)
		t = a[14] ^ d4
		bc0 = bits.RotateLeft64(t, 27)
		a[20] = bc0 ^ (bc2 &^ bc1)
		a[6] = bc1 ^ (bc3 &^ bc2)
		a[17] = bc2 ^ (bc4 &^ bc3)
		a[3] = bc3 ^ (bc0 &^ bc4)
		a[14] = bc4 ^ (bc1 &^ bc0)

		t = a[10] ^ d0
		bc3 = bits.RotateLeft64(t, 41)
		t = a[21] ^ d1
		bc4 = bits.RotateLeft64(t, 2)
		t = a[7] ^ d2
		bc0 = bits.RotateLeft64(t, 62)
		t = a[18] ^ d3
		bc1 = bits.RotateLeft64(t, 55)
		t = a[4] ^ d4
		bc2 = bits.RotateLeft64(t, 39)
		a[10] = bc0 ^ (bc2 &^ bc1)
		a[21] = bc1 ^ (bc3 &^ bc2)
		a[7] = bc2 ^ (bc4 &^ bc3)
		a[18] = bc3 ^ (bc0 &^ bc4)
		a[4] = bc4 ^ (bc1 &^ bc0)

		// Round 4
		bc0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20]
		bc1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21]
		bc2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22]
		bc3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23]
		bc4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24]
		d0 = bc4 ^ bits.RotateLeft64(bc1, 1)
		d1 = bc0 ^ bits.RotateLeft64(bc2, 1)
		d2 = bc1 ^ bits.RotateLeft64(bc3, 1)
		d3 = bc2 ^ bits.RotateLeft64(bc4, 1)
		d4 = bc3 ^ bits.RotateLeft64(bc0, 1)

		bc0 = a[0] ^ d0
		t = a[1] ^ d1
		bc1 = bits.RotateLeft64(t, 44)
		t = a[2] ^ d2
		bc2 = bits.RotateLeft64(t, 43)
		t = a[3] ^ d3
		bc3 = bits.RotateLeft64(t, 21)
		t = a[4] ^ d4
		bc4 = bits.RotateLeft64(t, 14)
		a[0] = bc0 ^ (bc2 &^ bc1) ^ roundConstants[i+3]
		a[1] = bc1 ^ (bc3 &^ bc2)
		a[2] = bc2 ^ (bc4 &^ bc3)
		a[3] = bc3 ^ (bc0 &^ bc4)
		a[4] = bc4 ^ (bc1 &^ bc0)

		t = a[5] ^ d0
		bc2 = bits.RotateLeft64(t, 3)
		t = a[6] ^ d1
		bc3 = bits.RotateLeft64(t, 45)
		t = a[7] ^ d2
		bc4 = bits.RotateLeft64(t, 61)
		t = a[8] ^ d3
		bc0 = bits.RotateLeft64(t, 28)
		t = a[9] ^ d4
		bc1 = bits.RotateLeft64(t, 20)
		a[5] = bc0 ^ (bc2 &^ bc1)
		a[6] = bc1 ^ (bc3 &^ bc2)
		a[7] = bc2 ^ (bc4 &^ bc3)
		a[8] = bc3 ^ (bc0 &^ bc4)
		a[9] = bc4 ^ (bc1 &^ bc0)

		t = a[10] ^ d0
		bc4 = bits.RotateLeft64(t, 18)
		t = a[11] ^ d1
		bc0 = bits.RotateLeft64(t, 1)
		t = a[12] ^ d2
		bc1 = bits.RotateLeft64(t, 6)
		t = a[13] ^ d3
		bc2 = bits.RotateLeft64(t, 25)
		t = a[14] ^ d4
		bc3 = bits.RotateLeft64(t, 8)
		a[10] = bc0 ^ (bc2 &^ bc1)
		a[11] = bc1 ^ (bc3 &^ bc2)
		a[12] = bc2 ^ (bc4 &^ bc3)
		a[13] = bc3 ^ (bc0 &^ bc4)
		a[14] = bc4 ^ (bc1 &^ bc0)

		t = a[15] ^ d0
		bc1 = bits.RotateLeft64(t, 36)
		t = a[16] ^ d1
		bc2 = bits.RotateLeft64(t, 10)
		t = a[17] ^ d2
		bc3 = bits.RotateLeft64(t, 15)
		t = a[18] ^ d3
		bc4 = bits.RotateLeft64(t, 56)
		t = a[19] ^ d4
		bc0 = bits.RotateLeft64(t, 27)
		a[15] = bc0 ^ (bc2 &^ bc1)
		a[16] = bc1 ^ (bc3 &^ bc2)
		a[17] = bc2 ^ (bc4 &^ bc3)
		a[18] = bc3 ^ (bc0 &^ bc4)
		a[19] = bc4 ^ (bc1 &^ bc0)

		t = a[20] ^ d0
		bc3 = bits.RotateLeft64(t, 41)
		t = a[21] ^ d1
		bc4 = bits.RotateLeft64(t, 2)
		t = a[22] ^ d2
		bc0 = bits.RotateLeft64(t, 62)
		t = a[23] ^ d3
		bc1 = bits.RotateLeft64(t, 55)
		t = a[24] ^ d4
		bc2 = bits.RotateLeft64(t, 39)
		a[20] = bc0 ^ (bc2 &^ bc1)
		a[21] = bc1 ^ (bc3 &^ bc2)
		a[22] = bc2 ^ (bc4 &^ bc3)
		a[23] = bc3 ^ (bc0 &^ bc4)
		a[24] = bc4 ^ (bc1 &^ bc0)
	}
}

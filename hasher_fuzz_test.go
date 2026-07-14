package kt128

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// FuzzHasher is an end-to-end differential fuzz of the public Hasher against a
// self-contained KT128 reference that routes exclusively through the pure-Go
// keccakP1600x12 permutation. On amd64/arm64 this pits the full assembly path
// (single-lane permute, fused absorb loop, and the SIMD leaf kernels reached in
// tree mode) against an asm-free oracle; on a purego build it still cross-checks
// the streaming and tree logic against an independent implementation.
//
// The fuzzer varies message content, customization string, write chunking, and
// output length, so it exercises the buffering, S_0 straddling, batch, and
// remainder paths far more adversarially than a fixed corpus.
func FuzzHasher(f *testing.F) {
	f.Add([]byte(""), []byte(""), uint16(1), uint16(32))
	f.Add([]byte("hello, world"), []byte(""), uint16(1), uint16(32))
	f.Add([]byte("msg"), []byte("custom"), uint16(2), uint16(64))
	// 8192-byte message with an empty customization string: S is 8193 bytes, so
	// this is the smallest input that enters tree mode (one trailing leaf).
	f.Add(bytes.Repeat([]byte{0xA5}, BlockSize), []byte(""), uint16(7), uint16(64))
	// Full single-node chunk written in one shot.
	f.Add(bytes.Repeat([]byte{0x5A}, BlockSize-1), []byte(""), uint16(BlockSize), uint16(32))
	// Multi-batch tree mode (9 chunks) with a customization string: enough leaves
	// to trigger one full 8-wide SIMD batch.
	f.Add(bytes.Repeat([]byte{0xCD}, 9*BlockSize), []byte("domain"), uint16(168), uint16(137))
	// Tree mode with a SIMD-batch remainder, split on a sub-chunk boundary.
	f.Add(bytes.Repeat([]byte{0x3C}, 11*BlockSize+123), []byte(""), uint16(4096), uint16(256))
	// Large customization string so S_0 straddles message and suffix.
	f.Add([]byte("m"), bytes.Repeat([]byte{0x11}, BlockSize+64), uint16(3), uint16(96))

	f.Fuzz(func(t *testing.T, msg, custom []byte, chunkRaw, outRaw uint16) {
		// Bound sizes so iterations stay fast while still spanning chunk and
		// SIMD-batch boundaries. 96 KiB reaches a full 8-wide leaf batch plus a
		// remainder; chunk in [1, 8193] guarantees large inputs are split across
		// many Write calls; outLen in [1, 4096] spans many squeeze blocks (rate
		// is 168). Without the cap the fuzzer balloons inputs until chunk=1 over a
		// huge message dominates each iteration and throughput collapses.
		const maxLen = 96 * 1024
		if len(msg) > maxLen {
			msg = msg[:maxLen]
		}
		if len(custom) > maxLen {
			custom = custom[:maxLen]
		}
		chunk := int(chunkRaw)%(BlockSize+1) + 1
		outLen := int(outRaw)%4096 + 1

		h := New(custom)
		for off := 0; off < len(msg); off += chunk {
			if _, err := h.Write(msg[off:min(off+chunk, len(msg))]); err != nil {
				t.Fatalf("Write: %v", err)
			}
		}
		got := make([]byte, outLen)
		if _, err := h.Read(got); err != nil {
			t.Fatalf("Read: %v", err)
		}

		want := referenceKT128(msg, custom, outLen)
		if !bytes.Equal(got, want) {
			t.Fatalf("KT128 mismatch (msg=%d custom=%d chunk=%d out=%d)\n got  %x\n want %x",
				len(msg), len(custom), chunk, outLen, got, want)
		}
	})
}

// referenceKT128 computes KT128(msg, custom) truncated to outLen bytes using
// only keccakP1600x12. It is a direct transcription of RFC 9861 §2 and shares no
// code with the production hasher beyond the (independently validated) pure-Go
// permutation.
func referenceKT128(msg, custom []byte, outLen int) []byte {
	// S = msg || custom || length_encode(|custom|).
	s := make([]byte, 0, len(msg)+len(custom)+9)
	s = append(s, msg...)
	s = append(s, custom...)
	s = append(s, refLengthEncode(uint64(len(custom)))...)

	// Single node when S fits in one chunk.
	if len(s) <= BlockSize {
		return refTurboSHAKE128(s, singleDS, outLen)
	}

	// Tree mode: final node = S_0 || 110^62 || CV_1 || ... || CV_n ||
	//            length_encode(n) || 0xFF 0xFF, hashed with the tree domain byte.
	var final refSponge
	final.absorb(s[:BlockSize])
	final.absorb([]byte{0x03, 0, 0, 0, 0, 0, 0, 0}) // 110^62 marker

	var leafCount uint64
	for rest := s[BlockSize:]; len(rest) > 0; {
		n := min(len(rest), BlockSize)
		cv := refTurboSHAKE128(rest[:n], leafDS, 32) // 256-bit leaf chain value
		final.absorb(cv)
		leafCount++
		rest = rest[n:]
	}

	final.absorb(refLengthEncode(leafCount))
	final.absorb([]byte{0xFF, 0xFF})
	final.padPermute(treeDS)

	out := make([]byte, outLen)
	final.squeeze(out)
	return out
}

// refTurboSHAKE128 is TurboSHAKE128(data, ds) truncated to outLen bytes.
func refTurboSHAKE128(data []byte, ds byte, outLen int) []byte {
	var s refSponge
	s.absorb(data)
	s.padPermute(ds)
	out := make([]byte, outLen)
	s.squeeze(out)
	return out
}

// refLengthEncode returns the KangarooTwelve length encoding of x: the bytes of
// x big-endian with no leading zeros, followed by a byte giving their count.
func refLengthEncode(x uint64) []byte {
	if x == 0 {
		return []byte{0}
	}
	var be [8]byte
	binary.BigEndian.PutUint64(be[:], x)
	i := 0
	for be[i] == 0 {
		i++
	}
	return append(bytes.Clone(be[i:]), byte(8-i))
}

// refSponge is a minimal, byte-at-a-time Keccak sponge built only on
// keccakP1600x12. It is intentionally simple rather than fast.
type refSponge struct {
	a   [lanes]uint64
	pos int // bytes absorbed into the current block, in [0, rate)
}

func (s *refSponge) absorb(p []byte) {
	for _, b := range p {
		s.a[s.pos>>3] ^= uint64(b) << (8 * (s.pos & 7))
		s.pos++
		if s.pos == rate {
			keccakP1600x12(&s.a)
			s.pos = 0
		}
	}
}

func (s *refSponge) padPermute(ds byte) {
	s.a[s.pos>>3] ^= uint64(ds) << (8 * (s.pos & 7))
	s.a[(rate-1)>>3] ^= uint64(0x80) << (8 * ((rate - 1) & 7))
	keccakP1600x12(&s.a)
	s.pos = 0
}

func (s *refSponge) squeeze(out []byte) {
	for i := range out {
		if s.pos == rate {
			keccakP1600x12(&s.a)
			s.pos = 0
		}
		out[i] = byte(s.a[s.pos>>3] >> (8 * (s.pos & 7)))
		s.pos++
	}
}

package kt128

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

func TestXOFMarshalBinaryStableEncoding(t *testing.T) {
	h := &XOF{
		state: state{
			c:     []byte("ctx"),
			pos:   0x0102030405060708,
			phase: stateFinalized,
			final: sponge{
				a:   [lanes]uint64{0: 0x0807060504030201, lanes - 1: 0x1122334455667788},
				pos: rate,
			},
		},
	}

	got, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	want := []byte("kt128")
	want = append(want, hashStateVersion, stateKindXOF, stateFinalized)
	want = binary.BigEndian.AppendUint64(want, 0x0102030405060708)
	want = binary.BigEndian.AppendUint64(want, 3)
	want = append(want, 1, 2, 3, 4, 5, 6, 7, 8)
	want = append(want, make([]byte, (lanes-2)*8)...)
	want = append(want, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11)
	want = append(want, rate)
	want = append(want, make([]byte, lanes*8)...)
	want = append(want, 0)
	want = append(want, "ctx"...)

	if !bytes.Equal(got, want) {
		t.Fatalf("MarshalBinary() encoding changed\n got  %x\n want %x", got, want)
	}
}

func TestXOFMarshalBinaryAbsorbingRoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		msgLen int
		custom []byte
		step   int
	}{
		{"empty", 0, nil, 1},
		{"single byte", 1, []byte("domain"), 1},
		{"rate boundary", rate, ptn(31), 37},
		{"last single-node byte", ChunkSize, nil, 113},
		{"first tree byte", ChunkSize + 1, []byte("tree"), 167},
		{"completed tree leaf", 2 * ChunkSize, nil, ChunkSize - 1},
		{"partial tree leaf", 3*ChunkSize + rate + 7, ptn(19), 1000},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			msg := ptn(tc.msgLen)
			h := NewXOF(tc.custom)
			for off := 0; off < len(msg); off += tc.step {
				_, _ = h.Write(msg[off:min(off+tc.step, len(msg))])
			}

			encoded, err := h.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			restored := NewXOF([]byte("replaced customization"))
			_, _ = restored.Write([]byte("replaced state"))
			if err := restored.UnmarshalBinary(encoded); err != nil {
				t.Fatal(err)
			}
			if restored.Pos() != uint64(len(msg)) {
				t.Fatalf("Pos() = %d, want %d", restored.Pos(), len(msg))
			}
			remarshaled, err := restored.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(remarshaled, encoded) {
				t.Fatal("restored state did not marshal canonically")
			}

			continuation := ptn(257)
			_, _ = restored.Write(continuation)
			_, _ = h.Write(continuation)
			got := make([]byte, 513)
			original := make([]byte, len(got))
			_, _ = restored.Read(got)
			_, _ = h.Read(original)
			full := append(bytes.Clone(msg), continuation...)
			if want := referenceKT128(full, tc.custom, len(got)); !bytes.Equal(got, want) {
				t.Fatalf("continued output mismatch\n got  %x\n want %x", got, want)
			}
			if !bytes.Equal(original, got) {
				t.Fatal("MarshalBinary modified the original absorbing state")
			}
		})
	}
}

func TestHashMarshalBinaryRoundTrip(t *testing.T) {
	custom := []byte("domain")
	h := NewHash(custom)
	_, _ = h.Write(ptn(ChunkSize + 17))

	encoded, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if encoded[marshalKindOffset] != stateKindHash {
		t.Fatalf("state kind = %d, want Hash", encoded[marshalKindOffset])
	}

	var restored Hash
	if err := restored.UnmarshalBinary(encoded); err != nil {
		t.Fatal(err)
	}
	if restored.Size() != DigestSize {
		t.Fatalf("restored Size() = %d, want %d", restored.Size(), DigestSize)
	}
	continuation := []byte("continuation")
	_, _ = h.Write(continuation)
	_, _ = restored.Write(continuation)
	if got, want := restored.Sum(nil), h.Sum(nil); !bytes.Equal(got, want) {
		t.Fatal("restored Hash diverged after continuation")
	}
}

func TestMarshalBinaryRejectsWrongConcreteType(t *testing.T) {
	hashEncoding, err := NewHash(nil).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	xofEncoding, err := NewXOF(nil).MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	var h Hash
	if err := h.UnmarshalBinary(xofEncoding); err == nil {
		t.Fatal("Hash accepted XOF state")
	}
	var x XOF
	if err := x.UnmarshalBinary(hashEncoding); err == nil {
		t.Fatal("XOF accepted Hash state")
	}
}

func TestXOFMarshalBinaryFinalizedRoundTrip(t *testing.T) {
	for _, msgLen := range []int{37, 2*ChunkSize + 19} {
		for _, squeezed := range []int{0, 1, rate - 1, rate, rate + 1, 3*rate + 9} {
			name := fmt.Sprintf("message=%d/squeezed=%d", msgLen, squeezed)
			t.Run(name, func(t *testing.T) {
				custom := []byte("domain")
				msg := ptn(msgLen)
				h := NewXOF(custom)
				_, _ = h.Write(msg)
				_, _ = h.Read(make([]byte, squeezed))

				encoded, err := h.MarshalBinary()
				if err != nil {
					t.Fatal(err)
				}
				var restored XOF
				if err := restored.UnmarshalBinary(encoded); err != nil {
					t.Fatal(err)
				}

				got := make([]byte, 511)
				original := make([]byte, len(got))
				_, _ = restored.Read(got)
				_, _ = h.Read(original)
				full := referenceKT128(msg, custom, squeezed+len(got))
				if want := full[squeezed:]; !bytes.Equal(got, want) {
					t.Fatalf("continued squeeze mismatch\n got  %x\n want %x", got, want)
				}
				if !bytes.Equal(original, got) {
					t.Fatal("MarshalBinary modified the original finalized state")
				}

				restored.Reset()
				resetMsg := []byte("after reset")
				_, _ = restored.Write(resetMsg)
				resetOut := make([]byte, 64)
				_, _ = restored.Read(resetOut)
				if want := referenceKT128(resetMsg, custom, len(resetOut)); !bytes.Equal(resetOut, want) {
					t.Fatal("Reset did not preserve the unmarshaled customization")
				}
			})
		}
	}
}

func TestXOFAppendBinary(t *testing.T) {
	h := NewXOF([]byte("custom"))
	_, _ = h.Write(ptn(ChunkSize + 17))

	marshaled, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	prefix := []byte{0xaa, 0xbb, 0xcc}
	b := make([]byte, len(prefix), len(prefix)+len(marshaled))
	copy(b, prefix)
	got, err := h.AppendBinary(b)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got[:len(prefix)], prefix) {
		t.Fatalf("AppendBinary modified prefix: %x", got[:len(prefix)])
	}
	if !bytes.Equal(got[len(prefix):], marshaled) {
		t.Fatal("MarshalBinary is not equivalent to AppendBinary(nil)")
	}
}

func TestXOFMarshalBinaryCanonicalAcrossChunking(t *testing.T) {
	custom := ptn(29)
	msg := ptn(9*ChunkSize + rate + 13)

	bulk := NewXOF(custom)
	_, _ = bulk.Write(msg)
	incremental := NewXOF(custom)
	for off := 0; off < len(msg); off += 137 {
		_, _ = incremental.Write(msg[off:min(off+137, len(msg))])
	}

	bulkEncoding, err := bulk.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	incrementalEncoding, err := incremental.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bulkEncoding, incrementalEncoding) {
		t.Fatal("absorbing encoding depends on Write chunking")
	}

	_, _ = bulk.Read(make([]byte, 2*rate+19))
	for _, n := range []int{1, rate - 1, rate, 19} {
		_, _ = incremental.Read(make([]byte, n))
	}
	bulkEncoding, err = bulk.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	incrementalEncoding, err = incremental.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bulkEncoding, incrementalEncoding) {
		t.Fatal("finalized encoding depends on Read chunking")
	}
}

func TestXOFUnmarshalBinaryCopiesCustomization(t *testing.T) {
	custom := []byte("owned customization")
	h := NewXOF(custom)
	encoded, err := h.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	var restored XOF
	if err := restored.UnmarshalBinary(encoded); err != nil {
		t.Fatal(err)
	}
	clear(encoded[marshaledStateSize:])

	restored.Reset()
	msg := []byte("message")
	_, _ = restored.Write(msg)
	got := make([]byte, 64)
	_, _ = restored.Read(got)
	if want := referenceKT128(msg, custom, len(got)); !bytes.Equal(got, want) {
		t.Fatal("UnmarshalBinary retained its input")
	}
}

func TestXOFUnmarshalBinaryRejectsInvalidStatesAtomically(t *testing.T) {
	single := NewXOF([]byte("custom"))
	_, _ = single.Write(ptn(100))
	singleEncoding, _ := single.MarshalBinary()

	tree := NewXOF(nil)
	_, _ = tree.Write(ptn(2*ChunkSize + 17))
	treeEncoding, _ := tree.MarshalBinary()

	completedTree := NewXOF(nil)
	_, _ = completedTree.Write(ptn(2 * ChunkSize))
	completedTreeEncoding, _ := completedTree.MarshalBinary()

	finalized := NewXOF(nil)
	_, _ = finalized.Read(nil)
	finalizedEncoding, _ := finalized.MarshalBinary()

	clone := bytes.Clone
	putPos := func(b []byte, pos uint64) { binary.BigEndian.PutUint64(b[marshalPosOffset:], pos) }
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"short identifier", []byte("kt12")},
		{"bad identifier", func() []byte { b := clone(singleEncoding); b[0] ^= 1; return b }()},
		{"bad version", func() []byte { b := clone(singleEncoding); b[marshalVersionOffset]++; return b }()},
		{"bad kind", func() []byte { b := clone(singleEncoding); b[marshalKindOffset] = stateKindHash; return b }()},
		{"bad lifecycle", func() []byte { b := clone(singleEncoding); b[marshalStateOffset] = 0xff; return b }()},
		{"wrong customization length", func() []byte {
			b := clone(singleEncoding)
			binary.BigEndian.PutUint64(b[marshalCustomLenOffset:], uint64(len(single.c)+1))
			return b
		}()},
		{"trailing data", append(clone(singleEncoding), 0)},
		{"single position in tree range", func() []byte {
			b := clone(singleEncoding)
			putPos(b, ChunkSize+1)
			return b
		}()},
		{"single final position mismatch", func() []byte {
			b := clone(singleEncoding)
			b[marshalFinalOffset+lanes*8]++
			return b
		}()},
		{"single active leaf", func() []byte { b := clone(singleEncoding); b[marshalLeafOffset] = 1; return b }()},
		{"single nonzero untouched byte", func() []byte {
			b := clone(singleEncoding)
			b[marshalFinalOffset+100] = 1
			return b
		}()},
		{"tree position in single range", func() []byte {
			b := clone(treeEncoding)
			putPos(b, ChunkSize)
			return b
		}()},
		{"tree final position mismatch", func() []byte {
			b := clone(treeEncoding)
			b[marshalFinalOffset+lanes*8]++
			return b
		}()},
		{"tree leaf position mismatch", func() []byte {
			b := clone(treeEncoding)
			b[marshalLeafOffset+lanes*8]++
			return b
		}()},
		{"tree nonzero untouched leaf byte", func() []byte {
			b := clone(treeEncoding)
			b[marshalLeafOffset+17] = 1
			return b
		}()},
		{"completed tree retained leaf", func() []byte {
			b := clone(completedTreeEncoding)
			b[marshalLeafOffset] = 1
			return b
		}()},
		{"absorbing final position out of range", func() []byte {
			b := clone(singleEncoding)
			b[marshalFinalOffset+lanes*8] = rate
			return b
		}()},
		{"leaf position out of range", func() []byte {
			b := clone(treeEncoding)
			b[marshalLeafOffset+lanes*8] = rate
			return b
		}()},
		{"finalized active leaf", func() []byte {
			b := clone(finalizedEncoding)
			b[marshalLeafOffset] = 1
			return b
		}()},
		{"finalized leaf position", func() []byte {
			b := clone(finalizedEncoding)
			b[marshalLeafOffset+lanes*8] = 1
			return b
		}()},
		{"finalized position out of range", func() []byte {
			b := clone(finalizedEncoding)
			b[marshalFinalOffset+lanes*8] = rate + 1
			return b
		}()},
	}
	for n := range len(singleEncoding) {
		tests = append(tests, struct {
			name string
			data []byte
		}{fmt.Sprintf("truncated at %d", n), singleEncoding[:n]})
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			target := NewXOF([]byte("existing"))
			_, _ = target.Write(ptn(ChunkSize + 3))
			before, err := target.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			input := bytes.Clone(tc.data)
			if err := target.UnmarshalBinary(input); err == nil {
				t.Fatal("UnmarshalBinary accepted invalid state")
			}
			if !bytes.Equal(input, tc.data) {
				t.Fatal("UnmarshalBinary modified its input")
			}
			after, err := target.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(after, before) {
				t.Fatal("UnmarshalBinary modified receiver after an error")
			}
		})
	}
}

func TestXOFMarshalBinaryRejectsInvalidInternalState(t *testing.T) {
	tests := []struct {
		name string
		h    XOF
	}{
		{"unknown lifecycle", XOF{state: state{phase: 99}}},
		{"single position", XOF{state: state{pos: 1, phase: stateSingle}}},
		{"single leaf", XOF{state: state{phase: stateSingle, leaf: sponge{a: [lanes]uint64{1}}}}},
		{"tree range", XOF{state: state{phase: stateTree}}},
		{"tree leaf length", XOF{state: state{pos: ChunkSize + 1, phase: stateTree, final: sponge{pos: treeFinalPosition(ChunkSize + 1)}}}},
		{"finalized leaf", XOF{state: state{phase: stateFinalized, leafLen: 1}}},
		{"finalized position", XOF{state: state{phase: stateFinalized, final: sponge{pos: rate + 1}}}},
	}
	for i := range tests {
		tc := &tests[i]
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.h.MarshalBinary(); err == nil {
				t.Fatal("MarshalBinary accepted invalid internal state")
			}
		})
	}
}

func FuzzXOFUnmarshalBinary(f *testing.F) {
	for _, size := range []int{0, 100, ChunkSize + 1, 2*ChunkSize + rate} {
		h := NewXOF([]byte("custom"))
		_, _ = h.Write(ptn(size))
		encoded, _ := h.MarshalBinary()
		f.Add(encoded)
	}
	h := NewXOF([]byte("custom"))
	_, _ = h.Write(ptn(100))
	_, _ = h.Read(make([]byte, rate+3))
	encoded, _ := h.MarshalBinary()
	f.Add(encoded)

	f.Fuzz(func(t *testing.T, data []byte) {
		var h1 XOF
		if err := h1.UnmarshalBinary(data); err != nil {
			return
		}
		canonical, err := h1.MarshalBinary()
		if err != nil {
			t.Fatalf("accepted state failed to marshal: %v", err)
		}
		if !bytes.Equal(canonical, data) {
			t.Fatal("accepted state was not canonical")
		}

		var h2 XOF
		if err := h2.UnmarshalBinary(canonical); err != nil {
			t.Fatalf("canonical state failed to unmarshal: %v", err)
		}
		if h1.phase != stateFinalized {
			_, _ = h1.Write([]byte("continuation"))
			_, _ = h2.Write([]byte("continuation"))
		}
		out1, out2 := make([]byte, 257), make([]byte, 257)
		_, _ = h1.Read(out1)
		_, _ = h2.Read(out2)
		if !bytes.Equal(out1, out2) {
			t.Fatal("restored states diverged")
		}
	})
}

func FuzzHashUnmarshalBinary(f *testing.F) {
	for _, messageSize := range []int{0, 100, ChunkSize + 1} {
		h := NewHash([]byte("custom"))
		_, _ = h.Write(ptn(messageSize))
		encoded, _ := h.MarshalBinary()
		f.Add(encoded)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		var h1 Hash
		if err := h1.UnmarshalBinary(data); err != nil {
			return
		}
		canonical, err := h1.MarshalBinary()
		if err != nil {
			t.Fatalf("accepted state failed to marshal: %v", err)
		}
		if !bytes.Equal(canonical, data) {
			t.Fatal("accepted state was not canonical")
		}

		var h2 Hash
		if err := h2.UnmarshalBinary(canonical); err != nil {
			t.Fatalf("canonical state failed to unmarshal: %v", err)
		}
		_, _ = h1.Write([]byte("continuation"))
		_, _ = h2.Write([]byte("continuation"))
		if !bytes.Equal(h1.Sum(nil), h2.Sum(nil)) {
			t.Fatal("restored states diverged")
		}
	})
}

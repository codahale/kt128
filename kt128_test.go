package kt128

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"io"
	"strings"
	"testing"
)

// ptn returns a byte slice of length n using the KT128 test pattern:
// repeating 0x00..0xFA (251 bytes).
func ptn(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i % 251)
	}
	return b
}

// RFC 9861 Section 5 KT128 test vectors.
var rfcVectors = []struct {
	name   string
	msg    []byte
	custom []byte
	outLen int
	want   []byte // full output (or last 32 bytes for 10032 case)
	last32 bool   // if true, want is the last 32 bytes of outLen output
}{
	{
		name:   "empty/empty/32",
		msg:    nil,
		custom: nil,
		outLen: 32,
		want:   mustHex("1AC2D450FC3B4205D19DA7BFCA1B37513C0803577AC7167F06FE2CE1F0EF39E5"),
	},
	{
		name:   "empty/empty/64",
		msg:    nil,
		custom: nil,
		outLen: 64,
		want: mustHex("1AC2D450FC3B4205D19DA7BFCA1B37513C0803577AC7167F06FE2CE1F0EF39E5" +
			"4269C056B8C82E48276038B6D292966CC07A3D4645272E31FF38508139EB0A71"),
	},
	{
		name:   "empty/empty/10032",
		msg:    nil,
		custom: nil,
		outLen: 10032,
		want:   mustHex("E8DC563642F7228C84684C898405D3A834799158C079B12880277A1D28E2FF6D"),
		last32: true,
	},
	{
		name:   "ptn(1)/empty/32",
		msg:    ptn(1),
		custom: nil,
		outLen: 32,
		want:   mustHex("2BDA92450E8B147F8A7CB629E784A058EFCA7CF7D8218E02D345DFAA65244A1F"),
	},
	{
		name:   "ptn(17)/empty/32",
		msg:    ptn(17),
		custom: nil,
		outLen: 32,
		want:   mustHex("6BF75FA2239198DB4772E36478F8E19B0F371205F6A9A93A273F51DF37122888"),
	},
	{
		name:   "ptn(289)/empty/32",
		msg:    ptn(289),
		custom: nil,
		outLen: 32,
		want:   mustHex("0C315EBCDEDBF61426DE7DCF8FB725D1E74675D7F5327A5067F367B108ECB67C"),
	},
	{
		name:   "ptn(4913)/empty/32",
		msg:    ptn(4913),
		custom: nil,
		outLen: 32,
		want:   mustHex("CB552E2EC77D9910701D578B457DDF772C12E322E4EE7FE417F92C758F0D59D0"),
	},
	{
		name:   "ptn(83521)/empty/32",
		msg:    ptn(83521),
		custom: nil,
		outLen: 32,
		want:   mustHex("8701045E22205345FF4DDA05555CBB5C3AF1A771C2B89BAEF37DB43D9998B9FE"),
	},
	{
		name:   "ptn(1419857)/empty/32",
		msg:    ptn(1419857),
		custom: nil,
		outLen: 32,
		want:   mustHex("844D610933B1B9963CBDEB5AE3B6B05CC7CBD67CEEDF883EB678A0A8E0371682"),
	},
	{
		name:   "ptn(24137569)/empty/32",
		msg:    ptn(24137569),
		custom: nil,
		outLen: 32,
		want:   mustHex("3C390782A8A4E89FA6367F72FEAAF13255C8D95878481D3CD8CE85F58E880AF8"),
	},
	{
		name:   "empty/ptn(1)/32",
		msg:    nil,
		custom: ptn(1),
		outLen: 32,
		want:   mustHex("FAB658DB63E94A246188BF7AF69A133045F46EE984C56E3C3328CAAF1AA1A583"),
	},
	{
		name:   "0xFF/ptn(41)/32",
		msg:    []byte{0xFF},
		custom: ptn(41),
		outLen: 32,
		want:   mustHex("D848C5068CED736F4462159B9867FD4C20B808ACC3D5BC48E0B06BA0A3762EC4"),
	},
	{
		name:   "0xFFx3/ptn(1681)/32",
		msg:    []byte{0xFF, 0xFF, 0xFF},
		custom: ptn(1681),
		outLen: 32,
		want:   mustHex("C389E5009AE57120854C2E8C64670AC01358CF4C1BAF89447A724234DC7CED74"),
	},
	{
		name:   "0xFFx7/ptn(68921)/32",
		msg:    []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
		custom: ptn(68921),
		outLen: 32,
		want:   mustHex("75D2F86A2E644566726B4FBCFC5657B9DBCF070C7B0DCA06450AB291D7443BCF"),
	},
	{
		name:   "ptn(8191)/empty/32",
		msg:    ptn(8191),
		custom: nil,
		outLen: 32,
		want:   mustHex("1B577636F723643E990CC7D6A659837436FD6A103626600EB8301CD1DBE553D6"),
	},
	{
		name:   "ptn(8192)/empty/32",
		msg:    ptn(8192),
		custom: nil,
		outLen: 32,
		want:   mustHex("48F256F6772F9EDFB6A8B661EC92DC93B95EBD05A08A17B39AE3490870C926C3"),
	},
	{
		name:   "ptn(8192)/ptn(8189)/32",
		msg:    ptn(8192),
		custom: ptn(8189),
		outLen: 32,
		want:   mustHex("3ED12F70FB05DDB58689510AB3E4D23C6C603384 9AA01E1D8C220A297FEDCD0B"),
	},
	{
		name:   "ptn(8192)/ptn(8190)/32",
		msg:    ptn(8192),
		custom: ptn(8190),
		outLen: 32,
		want:   mustHex("6A7C1B6A5CD0D8C9CA943A4A216CC646045 59A2EA45F78570A15253D67BA00AE"),
	},
}

func TestRFCVectors(t *testing.T) {
	for _, tc := range rfcVectors {
		t.Run(tc.name, func(t *testing.T) {
			h := New()
			h.SetCustomizationString(tc.custom)

			if tc.msg != nil {
				_, _ = h.Write(tc.msg)
			}

			out := make([]byte, tc.outLen)
			if len(tc.custom) > 0 {
				_, _ = h.Read(out)
			} else {
				_, _ = h.Read(out)
			}

			var got []byte
			if tc.last32 {
				got = out[len(out)-32:]
			} else {
				got = out
			}

			if !bytes.Equal(got, tc.want) {
				t.Errorf("got %x, want %x", got, tc.want)
			}
		})
	}
}

func TestClone(t *testing.T) {
	sizes := []int{0, 1, BlockSize - 1, BlockSize, BlockSize + 1, 83521}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%d", size), func(t *testing.T) {
			msg := ptn(size)

			// Write all data, clone, verify both produce the same output.
			h := New()
			_, _ = h.Write(msg)

			clone := h.Clone()

			// Use readCustom with a custom string to test clone + custom finalization.
			want := make([]byte, 64)
			_, _ = h.Read(want)

			got := make([]byte, 64)
			_, _ = clone.Read(got)

			if !bytes.Equal(got, want) {
				t.Errorf("size=%d: clone output mismatch", size)
			}
		})
	}

	t.Run("independent after clone", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(BlockSize + 1))

		clone := h.Clone()

		// Write more data to the original only.
		_, _ = h.Write([]byte("extra"))

		out1 := make([]byte, 64)
		_, _ = h.Read(out1)

		out2 := make([]byte, 64)
		_, _ = clone.Read(out2)

		if bytes.Equal(out1, out2) {
			t.Error("clone and original produced identical output after diverging")
		}
	})
}

func TestEqual(t *testing.T) {
	t.Run("same input", func(t *testing.T) {
		h1 := New()
		_, _ = h1.Write(ptn(100))

		h2 := New()
		_, _ = h2.Write(ptn(100))

		if h1.Equal(h2) != 1 {
			t.Fatal("identical hashers should be equal")
		}
	})

	t.Run("different input", func(t *testing.T) {
		h1 := New()
		_, _ = h1.Write(ptn(100))

		h2 := New()
		_, _ = h2.Write(ptn(200))

		if h1.Equal(h2) != 0 {
			t.Fatal("different hashers should not be equal")
		}
	})

	t.Run("clone", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(BlockSize + 1))

		clone := h.Clone()
		if h.Equal(clone) != 1 {
			t.Fatal("hasher and its clone should be equal")
		}
	})

	t.Run("diverged clone", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(100))

		clone := h.Clone()
		_, _ = clone.Write([]byte("extra"))

		if h.Equal(clone) != 0 {
			t.Fatal("diverged clone should not be equal")
		}
	})
}

func TestPos(t *testing.T) {
	t.Run("new hasher", func(t *testing.T) {
		h := New()
		if h.Pos() != 0 {
			t.Fatalf("Pos() = %d, want 0", h.Pos())
		}
	})

	t.Run("after write", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(100))
		if h.Pos() != 100 {
			t.Fatalf("Pos() = %d, want 100", h.Pos())
		}
	})

	t.Run("cumulative writes", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(100))
		_, _ = h.Write(ptn(200))
		if h.Pos() != 300 {
			t.Fatalf("Pos() = %d, want 300", h.Pos())
		}
	})

	t.Run("after reset", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(100))
		h.Reset()
		if h.Pos() != 0 {
			t.Fatalf("Pos() after Reset = %d, want 0", h.Pos())
		}
	})
}

func TestReset(t *testing.T) {
	h := New()
	_, _ = h.Write(ptn(BlockSize + 1))
	h.Reset()
	_, _ = h.Write(ptn(BlockSize + 1))

	fresh := New()
	_, _ = fresh.Write(ptn(BlockSize + 1))

	out1 := make([]byte, 64)
	_, _ = h.Read(out1)

	out2 := make([]byte, 64)
	_, _ = fresh.Read(out2)

	if !bytes.Equal(out1, out2) {
		t.Fatal("Reset hasher should produce same output as fresh hasher")
	}
}

func TestBlockSizeMethod(t *testing.T) {
	h := New()
	if got := h.BlockSize(); got != BlockSize {
		t.Fatalf("BlockSize() = %d, want %d", got, BlockSize)
	}
}

// TestLengthEncode checks lengthEncode against hand-computed golden values for
// RFC 9861 §2.3.1's own examples and every byte-width boundary. The leafCount and
// |C| encodings only reach the multi-byte forms on very large inputs, so the
// RFC vectors and the (size-capped) fuzzer barely exercise them; these golden
// values pin the encoding independently of any other implementation in the tree.
func TestLengthEncode(t *testing.T) {
	tests := []struct {
		value uint64
		want  []byte
	}{
		{0, []byte{0x00}}, // RFC 9861 example
		{1, []byte{0x01, 0x01}},
		{12, []byte{0x0C, 0x01}}, // RFC 9861 example
		{127, []byte{0x7F, 0x01}},
		{128, []byte{0x80, 0x01}},
		{255, []byte{0xFF, 0x01}},
		{256, []byte{0x01, 0x00, 0x02}},
		{65535, []byte{0xFF, 0xFF, 0x02}},
		{65536, []byte{0x01, 0x00, 0x00, 0x03}},
		{65538, []byte{0x01, 0x00, 0x02, 0x03}}, // RFC 9861 example
		{1<<24 - 1, []byte{0xFF, 0xFF, 0xFF, 0x03}},
		{1 << 24, []byte{0x01, 0x00, 0x00, 0x00, 0x04}},
		{1<<32 - 1, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0x04}},
		{1 << 32, []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x05}},
		{1 << 40, []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06}},
		{1 << 48, []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07}},
		{1 << 56, []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08}},
		{^uint64(0), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x08}},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%d", tc.value), func(t *testing.T) {
			if got := lengthEncode(nil, tc.value); !bytes.Equal(got, tc.want) {
				t.Errorf("lengthEncode(nil, %d) = %x, want %x", tc.value, got, tc.want)
			}

			// Callers always append onto a live buffer (the customization suffix
			// and the leafCount terminator), so a non-empty prefix must survive.
			prefix := []byte{0xAA, 0xBB}
			want := append([]byte{0xAA, 0xBB}, tc.want...)
			if got := lengthEncode(prefix, tc.value); !bytes.Equal(got, want) {
				t.Errorf("lengthEncode(prefix, %d) = %x, want %x", tc.value, got, want)
			}
		})
	}
}

// TestWritePartitionInvariance verifies that the output is independent of how the
// message is split across Write calls, across message and customization sizes
// that straddle chunk boundaries. This exercises the buffering and finalization
// paths far more densely than the RFC vectors.
func TestWritePartitionInvariance(t *testing.T) {
	// Sizes clustered around chunk and SIMD-batch boundaries.
	interesting := []int{
		0, 1, 2, 167, 168, 169,
		BlockSize - 2, BlockSize - 1, BlockSize, BlockSize + 1, BlockSize + 2,
		2*BlockSize - 1, 2 * BlockSize, 2*BlockSize + 1,
		7 * BlockSize, 8*BlockSize - 1, 8 * BlockSize, 8*BlockSize + 1,
		9 * BlockSize, 8*BlockSize + 168, 12345, 83521,
	}
	customs := []int{0, 1, 41, BlockSize - 4, BlockSize, BlockSize + 7, 2*BlockSize + 3}
	chunks := []int{1, 7, 168, 8191, BlockSize, BlockSize + 1, 3 * BlockSize}

	for _, msgLen := range interesting {
		msg := ptn(msgLen)
		for _, customLen := range customs {
			custom := ptn(customLen)

			// Reference: a single Write.
			ref := New()
			ref.SetCustomizationString(custom)
			_, _ = ref.Write(msg)
			want := make([]byte, 64)
			_, _ = ref.Read(want)

			for _, chunk := range chunks {
				h := New()
				h.SetCustomizationString(custom)
				for off := 0; off < len(msg); off += chunk {
					_, _ = h.Write(msg[off:min(off+chunk, len(msg))])
				}
				got := make([]byte, 64)
				_, _ = h.Read(got)
				if !bytes.Equal(got, want) {
					t.Fatalf("msgLen=%d customLen=%d chunk=%d: output depends on write partitioning",
						msgLen, customLen, chunk)
				}
			}
		}
	}
}

// TestReadPartitionInvariance verifies that XOF output is independent of how it
// is split across Read calls. A single Read squeezes lane-aligned, but resuming
// after a short Read leaves the sponge mid-lane, so this is the only thing that
// exercises squeeze's off != 0 branch and its permute-mid-Read path. outLen is
// neither a multiple of 8 nor of the rate (168), and the chunk sizes straddle
// both boundaries, so reads resume at every alignment.
func TestReadPartitionInvariance(t *testing.T) {
	const outLen = 1000

	// Single-node, chunk-boundary, and tree-mode messages give the final sponge
	// different contents to squeeze from.
	msgs := []int{0, 1, BlockSize - 1, BlockSize, BlockSize + 1, 9 * BlockSize}
	chunks := []int{1, 2, 3, 7, 8, 9, 167, 168, 169, 333}

	for _, msgLen := range msgs {
		msg := ptn(msgLen)

		// Reference: one Read of the whole output.
		ref := New()
		_, _ = ref.Write(msg)
		want := make([]byte, outLen)
		_, _ = ref.Read(want)

		for _, chunk := range chunks {
			h := New()
			_, _ = h.Write(msg)
			got := make([]byte, outLen)
			for off := 0; off < outLen; off += chunk {
				end := min(off+chunk, outLen)
				if _, err := h.Read(got[off:end]); err != nil {
					t.Fatalf("Read: %v", err)
				}
			}
			if !bytes.Equal(got, want) {
				t.Errorf("msgLen=%d chunk=%d: output depends on read partitioning", msgLen, chunk)
			}
		}
	}
}

func TestWriteTreeModeBuffering(t *testing.T) {
	t.Run("direct S0", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(BlockSize + 1))

		if h.state != stateTree {
			t.Fatalf("state = %d, want stateTree", h.state)
		}
		if len(h.buf) != 1 {
			t.Fatalf("buffered bytes = %d, want 1", len(h.buf))
		}
		if cap(h.buf) >= BlockSize {
			t.Fatalf("buffer capacity = %d, want less than one block", cap(h.buf))
		}
	})

	t.Run("reuse buffered S0", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(BlockSize))
		initialCap := cap(h.buf)
		_, _ = h.Write([]byte{0xA5})

		if h.state != stateTree {
			t.Fatalf("state = %d, want stateTree", h.state)
		}
		if len(h.buf) != 1 {
			t.Fatalf("buffered bytes = %d, want 1", len(h.buf))
		}
		if cap(h.buf) != initialCap {
			t.Fatalf("buffer capacity grew from %d to %d", initialCap, cap(h.buf))
		}
	})

	t.Run("flush exact lane batch", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn(BlockSize + 1))
		_, _ = h.Write(ptn(availableLanes*BlockSize - 1))

		if h.leafCount != uint64(availableLanes) {
			t.Fatalf("leaf count = %d, want %d", h.leafCount, availableLanes)
		}
		if len(h.buf) != 0 {
			t.Fatalf("buffered bytes = %d, want 0", len(h.buf))
		}
	})

	t.Run("process exact lane batch directly", func(t *testing.T) {
		h := New()
		_, _ = h.Write(ptn((availableLanes + 1) * BlockSize))

		if h.leafCount != uint64(availableLanes) {
			t.Fatalf("leaf count = %d, want %d", h.leafCount, availableLanes)
		}
		if cap(h.buf) != 0 {
			t.Fatalf("buffer capacity = %d, want 0", cap(h.buf))
		}
	})
}

func BenchmarkWrite(b *testing.B) {
	for _, size := range sizes {
		b.Run(size.Name, func(b *testing.B) {
			msg := ptn(size.N)
			out := make([]byte, 32)
			b.SetBytes(int64(size.N))
			b.ReportAllocs()
			for b.Loop() {
				h := New()
				_, _ = h.Write(msg)
				_, _ = h.Read(out)
			}
		})
	}
}

func BenchmarkWriteStreaming(b *testing.B) {
	for _, size := range sizes {
		if size.N < 2*BlockSize {
			continue
		}
		b.Run(size.Name, func(b *testing.B) {
			msg := ptn(size.N)
			out := make([]byte, 32)
			b.SetBytes(int64(size.N))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				h := New()
				for i := 0; i < len(msg); i += BlockSize {
					end := min(i+BlockSize, len(msg))
					_, _ = h.Write(msg[i:end])
				}
				_, _ = h.Read(out)
			}
		})
	}
}

func BenchmarkRead(b *testing.B) {
	for _, outSize := range []int{32, 64, 256, 1024} {
		b.Run(fmt.Sprintf("%d", outSize), func(b *testing.B) {
			out := make([]byte, outSize)
			b.SetBytes(int64(outSize))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				h := New()
				_, _ = h.Write(ptn(BlockSize + 1))
				_, _ = io.ReadFull(h, out)
			}
		})
	}
}

func mustHex(s string) []byte {
	s = strings.ReplaceAll(s, " ", "")
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

type size struct {
	Name string
	N    int
}

var sizes = []size{
	{"1B", 1},
	{"64B", 64},
	{"8KiB", 8 * 1024},
	{"8KiB+1B", BlockSize + 1},
	{"32KiB", 32 * 1024},
	{"64KiB", 64 * 1024},
	{"72KiB", 9 * BlockSize},
	{"1MiB", 1024 * 1024},
	{"16MiB", 16 * 1024 * 1024},
}

package kt128

import (
	"bytes"
	"testing"
)

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
			ref := New(custom)
			_, _ = ref.Write(msg)
			want := make([]byte, 64)
			_, _ = ref.Read(want)

			for _, chunk := range chunks {
				h := New(custom)
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
		ref := New(nil)
		_, _ = ref.Write(msg)
		want := make([]byte, outLen)
		_, _ = ref.Read(want)

		for _, chunk := range chunks {
			h := New(nil)
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

// TestWriteFusedS0Leaf checks that the fused S_0+leaf fast path (S_0 and the
// first leaf arriving in one Write of an untouched Hasher) produces the same
// output as the incremental path it bypasses.

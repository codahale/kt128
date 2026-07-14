// Package kt128 implements KT128 (KangarooTwelve) as specified in RFC 9861.
//
// KT128 is a tree-hash eXtendable-Output Function (XOF) built on TurboSHAKE128.
// When the input (the message plus the customization string and its length
// encoding) exceeds one 8192-byte chunk, it splits the input into chunks and
// computes a leaf chain value from each. On amd64 and arm64 the leaves are
// computed in parallel using SIMD-accelerated Keccak permutations; other
// targets and the purego build use a scalar fallback.
package kt128

import (
	"hash"
	"slices"
)

const (
	// BlockSize is the KT128 chunk size in bytes.
	BlockSize = 8192

	leafDS   = 0x0B
	treeDS   = 0x06
	singleDS = 0x07

	// Hasher lifecycle states.
	stateSingle    uint8 = 0 // absorbing, single-node (< 1 chunk seen)
	stateTree      uint8 = 1 // absorbing, tree mode (S_0 flushed)
	stateFinalized uint8 = 2 // finalized and squeezable
)

// Hasher is an incremental KT128 instance.
type Hasher struct {
	buf        []byte       // buffered leaf data (tree mode only)
	c          []byte       // owned copy of the customization string
	final      sponge       // final-node sponge state
	pending    pendingState // partially-absorbed trailing leaf from a fused first write
	pos        uint64       // total bytes written via Write
	leafCount  uint64       // total leaf CVs written to final so far
	pendingLen int          // bytes absorbed into pending; 0 = no pending leaf
	state      uint8        // lifecycle: stateSingle -> stateTree -> stateFinalized
	ds         byte         // KT128 customization byte for finalization (singleDS or treeDS)
}

// New returns a new Hasher with the given customization string. The Hasher
// retains a copy of c, so later mutations of the caller's slice do not affect
// the output. Pass nil for no customization.
func New(c []byte) *Hasher {
	return &Hasher{c: slices.Clone(c)}
}

// BlockSize returns the KT128 chunk size in bytes (8192). Writes need not be a
// multiple of this size.
func (h *Hasher) BlockSize() int {
	return BlockSize
}

// Pos returns the total number of bytes written via Write.
func (h *Hasher) Pos() uint64 {
	return h.pos
}

// Write absorbs message bytes. It panics if called after Read, which finalizes
// the hasher.

var _ hash.XOF = (*Hasher)(nil)

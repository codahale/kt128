// Package kt128 implements KT128 (KangarooTwelve) as specified in [RFC 9861].
//
// KT128 is a tree-hash eXtendable-Output Function (XOF) built on TurboSHAKE128.
// Create a [Hasher] with [New], absorb the message with [Hasher.Write], then
// read as much output as needed with [Hasher.Read]. The first Read finalizes
// the message; subsequent reads continue the same output stream.
//
// When the input (the message plus the customization string and its length
// encoding) exceeds one 8192-byte chunk, it splits the input into chunks and
// computes a leaf chain value from each. On amd64 and arm64 the leaves are
// computed in parallel using SIMD-accelerated Keccak permutations when the
// required CPU features are available; other targets and the purego build use
// a scalar fallback.
//
// [RFC 9861]: https://www.rfc-editor.org/rfc/rfc9861.html
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

// noCopy is recognized by go vet's copylocks analyzer.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}

// Hasher is an incremental KT128 instance. Its zero value is ready to use with
// no customization string.
//
// A Hasher must not be copied after first use. Use [Hasher.Clone] to create an
// independent copy. A Hasher is not safe for concurrent mutation.
type Hasher struct {
	noCopy     noCopy
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

// New returns a new Hasher using c as the KT128 customization string. It copies
// c, so the caller may modify or reuse c after New returns. Pass nil for no
// customization.
func New(c []byte) *Hasher {
	return &Hasher{c: slices.Clone(c)}
}

// BlockSize returns the KT128 chunk size in bytes. Write accepts inputs of any
// length, but chunk-aligned writes may be processed more efficiently.
func (h *Hasher) BlockSize() int {
	return BlockSize
}

// Pos returns the total number of message bytes accepted by [Hasher.Write]
// since construction or the last call to [Hasher.Reset] or [Hasher.Clear].
func (h *Hasher) Pos() uint64 {
	return h.pos
}

var _ hash.XOF = (*Hasher)(nil)

// Package kt128 implements KT128 (KangarooTwelve) as specified in [RFC 9861].
//
// KT128 is a tree-hash eXtendable-Output Function (XOF) built on TurboSHAKE128.
// Create a [Hasher] with [New], absorb the message with [Hasher.Write], then
// read as much output as needed with [Hasher.Read]. The first Read finalizes
// the message; subsequent reads continue the same output stream.
//
// When the input (the message plus the customization string and its length
// encoding) exceeds [ChunkSize] bytes, it splits the input into chunks and
// computes a leaf chain value from each. On amd64 and arm64 the leaves are
// computed in parallel using SIMD-accelerated Keccak permutations when the
// required CPU features are available; other targets and the purego build use
// a scalar fallback.
//
// # Security
//
// KT128 targets 128-bit security. Outputs should be at least 16 bytes for
// 128-bit single-target preimage and second-preimage resistance, and at least
// 32 bytes for 128-bit collision resistance. Multi-target preimage resistance
// may require additional output bits; see RFC 9861 Section 7.
//
// KT128 is designed to be fast and is not a password-hashing function. Use a
// purpose-built password-hashing function for passwords and other low-entropy
// secrets.
//
// [RFC 9861]: https://www.rfc-editor.org/rfc/rfc9861.html
package kt128

import "hash"

const (
	// ChunkSize is the KT128 chunk size in bytes.
	ChunkSize = 8192

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
// [New] retains its customization slice by reference. The caller must keep that
// slice's contents unchanged while the Hasher or any clone derived from it may
// be used.
// Write and Read do not retain their input or output slices.
//
// A Hasher must not be copied after first use. Use [Hasher.Clone] to create an
// independent copy. A Hasher is not safe for concurrent mutation.
type Hasher struct {
	noCopy  noCopy
	c       []byte // caller-owned customization string, retained by reference
	final   sponge // final-node sponge state
	leaf    sponge // current partial leaf (tree mode only)
	pos     uint64 // total bytes written via Write
	leafLen int    // bytes absorbed into leaf; 0 = no partial leaf
	state   uint8  // lifecycle: stateSingle -> stateTree -> stateFinalized
}

// New returns a new Hasher using c as the KT128 customization string. It retains
// c by reference without copying it. The caller must not modify c while the
// returned Hasher or any Hasher cloned from it may be used, and remains
// responsible for clearing c if necessary. Pass nil for no customization.
func New(c []byte) *Hasher {
	return &Hasher{c: c}
}

// BlockSize returns the 168-byte TurboSHAKE128 sponge rate. Write accepts inputs
// of any length; calls containing multiple complete chunks can use parallel leaf
// processing.
func (h *Hasher) BlockSize() int {
	return rate
}

// Pos returns the total number of message bytes accepted by [Hasher.Write]
// since construction or the last call to [Hasher.Reset]. Write panics before
// this count would reach 2^64 bytes.
func (h *Hasher) Pos() uint64 {
	return h.pos
}

var _ hash.XOF = (*Hasher)(nil)

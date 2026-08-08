package kt128

import (
	"encoding"
	"hash"
)

// Hash is an incremental KT128 hash with a 32-byte digest. Its zero value is
// ready to use with no customization string.
//
// A Hash must not be copied after first use. Use [Hash.Clone] to create an
// independent copy. A Hash is not safe for concurrent mutation.
type Hash struct{ state }

// NewHash returns a new Hash using c as the optional KT128 customization
// string.
//
// NewHash copies c; the caller may modify or clear c immediately after
// NewHash returns. Pass nil for no customization.
func NewHash(c []byte) *Hash {
	return &Hash{state: state{c: append([]byte(nil), c...)}}
}

func (h *Hash) core() *state { return &h.state }

// BlockSize returns the 168-byte TurboSHAKE128 sponge rate.
func (h *Hash) BlockSize() int { return rate }

// Size returns [DigestSize].
func (h *Hash) Size() int { return DigestSize }

// Pos returns the total number of message bytes accepted by [Hash.Write] since
// construction or the last call to [Hash.Reset].
func (h *Hash) Pos() uint64 { return h.pos }

// Write absorbs p as message data. It always returns len(p), nil and does not
// retain p. It panics if accepting p would bring the message length to 2^64
// bytes.
func (h *Hash) Write(p []byte) (int, error) { return h.core().write(p) }

// Sum appends the current 32-byte KT128 digest to b without changing h's state.
func (h *Hash) Sum(b []byte) []byte {
	var clone state
	h.core().cloneInto(&clone)
	size := h.Size()
	start := len(b)
	b = append(b, make([]byte, size)...)
	_, _ = clone.read(b[start:])
	return b
}

// Clone returns an independent copy of h. The returned error is always nil.
func (h *Hash) Clone() (hash.Cloner, error) {
	clone := new(Hash)
	h.core().cloneInto(clone.core())
	return clone, nil
}

// Reset reinitializes h for reuse with the same customization string. Reset
// does not guarantee erasure of the previous hashing state.
func (h *Hash) Reset() { h.core().reset() }

// MarshalBinary returns a type-tagged encoding of h.
func (h *Hash) MarshalBinary() ([]byte, error) {
	return h.AppendBinary(make([]byte, 0, marshaledStateSize+len(h.c)))
}

// AppendBinary appends the binary encoding of h to b. It does not modify
// b[:len(b)] or retain b.
func (h *Hash) AppendBinary(b []byte) ([]byte, error) {
	return appendState(b, h.core(), stateKindHash)
}

// UnmarshalBinary replaces h with the Hash state encoded by data. The receiver
// is unchanged if validation fails, and it does not retain data.
func (h *Hash) UnmarshalBinary(data []byte) error {
	decoded, err := unmarshalState(data, stateKindHash)
	if err != nil {
		return err
	}
	assignState(h.core(), decoded)
	return nil
}

var (
	_ encoding.BinaryMarshaler   = (*Hash)(nil)
	_ encoding.BinaryUnmarshaler = (*Hash)(nil)
	_ encoding.BinaryAppender    = (*Hash)(nil)
	_ hash.Cloner                = (*Hash)(nil)
	_ hash.Hash                  = (*Hash)(nil)
)

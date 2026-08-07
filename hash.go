package kt128

import (
	"encoding"
	"errors"
	"hash"
)

// Hash is an incremental KT128 hash with a fixed digest size. Its zero value is
// ready to use with a 32-byte digest and no customization string.
//
// A Hash must not be copied after first use. Use [Hash.Clone] to create an
// independent copy. A Hash is not safe for concurrent mutation.
type Hash struct{ state }

// NewHash returns a new fixed-digest Hash using c as the optional KT128
// customization string. It panics if digestSize is not positive.
//
// NewHash copies c; the caller may modify or clear c immediately after
// NewHash returns. Pass nil for no customization.
func NewHash(c []byte, digestSize int) *Hash {
	if digestSize <= 0 {
		panic("kt128: non-positive digest size")
	}
	return &Hash{state: state{c: append([]byte(nil), c...), digest: digestSize}}
}

func (h *Hash) core() *state { return &h.state }

// BlockSize returns the 168-byte TurboSHAKE128 sponge rate.
func (h *Hash) BlockSize() int { return rate }

// Size returns the number of bytes returned by [Hash.Sum].
func (h *Hash) Size() int {
	if h.digest == 0 {
		return defaultDigestSize
	}
	return h.digest
}

// Pos returns the total number of message bytes accepted by [Hash.Write] since
// construction or the last call to [Hash.Reset].
func (h *Hash) Pos() uint64 { return h.pos }

// Write absorbs p as message data. It always returns len(p), nil and does not
// retain p. It panics if accepting p would bring the message length to 2^64
// bytes.
func (h *Hash) Write(p []byte) (int, error) { return h.core().write(p) }

// Sum appends the current KT128 digest to b without changing h's state. The
// number of appended bytes is the configured digest size, or 32 for the zero
// value.
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

// Reset reinitializes h for reuse with the same customization string and
// digest size. Reset does not guarantee erasure of the previous hashing state.
func (h *Hash) Reset() { h.core().reset() }

// MarshalBinary returns a type-tagged encoding of h and its digest size.
func (h *Hash) MarshalBinary() ([]byte, error) {
	return h.AppendBinary(make([]byte, 0, marshaledStateSize+len(h.c)))
}

// AppendBinary appends the binary encoding of h to b. It does not modify
// b[:len(b)] or retain b.
func (h *Hash) AppendBinary(b []byte) ([]byte, error) {
	if h.digest < 0 {
		return nil, errors.New("kt128: invalid digest size")
	}
	return appendState(b, h.core(), stateKindHash, uint64(h.Size()))
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

package kt128

import (
	"encoding"
	"hash"
)

// XOF is an incremental KT128 extendable-output function. Its zero value is
// ready to use with no customization string.
//
// An XOF must not be copied after first use. Use [XOF.Clone] to create an
// independent copy. An XOF is not safe for concurrent mutation.
type XOF struct{ state }

// NewXOF returns a new XOF using c as the optional KT128 customization string.
//
// NewXOF copies c; the caller may modify or clear c immediately after NewXOF
// returns. Pass nil for no customization.
func NewXOF(c []byte) *XOF {
	return &XOF{state: state{c: append([]byte(nil), c...)}}
}

func (x *XOF) core() *state { return &x.state }

// BlockSize returns the 168-byte TurboSHAKE128 sponge rate.
func (x *XOF) BlockSize() int { return rate }

// Pos returns the total number of message bytes accepted by [XOF.Write] since
// construction or the last call to [XOF.Reset].
func (x *XOF) Pos() uint64 { return x.pos }

// Write absorbs p as message data. It always returns len(p), nil and does not
// retain p. It panics after any call to [XOF.Read], including a zero-length
// read, or if accepting p would bring the message length to 2^64 bytes.
func (x *XOF) Write(p []byte) (int, error) { return x.core().write(p) }

// Read fills p with output from the XOF and returns len(p), nil. The first call,
// including a zero-length call, finalizes the message; subsequent calls continue
// squeezing the same output stream. Read never returns io.EOF.
func (x *XOF) Read(p []byte) (int, error) { return x.core().read(p) }

// Clone returns an independent copy of x at its current absorption or squeeze
// position.
func (x *XOF) Clone() *XOF {
	clone := new(XOF)
	x.core().cloneInto(clone.core())
	return clone
}

// Reset reinitializes x for reuse with the same customization string. Reset
// does not guarantee erasure of the previous hashing state.
func (x *XOF) Reset() { x.core().reset() }

// MarshalBinary returns a type-tagged encoding of x at its current absorption
// or squeeze position.
func (x *XOF) MarshalBinary() ([]byte, error) {
	return x.AppendBinary(make([]byte, 0, marshaledStateSize+len(x.c)))
}

// AppendBinary appends the binary encoding of x to b. It does not modify
// b[:len(b)] or retain b.
func (x *XOF) AppendBinary(b []byte) ([]byte, error) {
	return appendState(b, x.core(), stateKindXOF, 0)
}

// UnmarshalBinary replaces x with the XOF state encoded by data. The receiver
// is unchanged if validation fails, and it does not retain data.
func (x *XOF) UnmarshalBinary(data []byte) error {
	decoded, err := unmarshalState(data, stateKindXOF)
	if err != nil {
		return err
	}
	assignState(x.core(), decoded)
	return nil
}

var (
	_ encoding.BinaryMarshaler   = (*XOF)(nil)
	_ encoding.BinaryUnmarshaler = (*XOF)(nil)
	_ encoding.BinaryAppender    = (*XOF)(nil)
	_ hash.XOF                   = (*XOF)(nil)
)

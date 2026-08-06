package kt128

import (
	"encoding"
	"encoding/binary"
	"errors"
)

const (
	hasherMarshalMagic   = "kt128"
	hasherMarshalVersion = 1

	marshaledSpongeSize = lanes*8 + 1
	marshaledStateSize  = len(hasherMarshalMagic) + 1 + 1 + 8 + 8 + 2*marshaledSpongeSize

	marshalVersionOffset   = len(hasherMarshalMagic)
	marshalStateOffset     = marshalVersionOffset + 1
	marshalPosOffset       = marshalStateOffset + 1
	marshalCustomLenOffset = marshalPosOffset + 8
	marshalFinalOffset     = marshalCustomLenOffset + 8
	marshalLeafOffset      = marshalFinalOffset + marshaledSpongeSize
)

var (
	_ encoding.BinaryMarshaler   = (*Hasher)(nil)
	_ encoding.BinaryUnmarshaler = (*Hasher)(nil)
	_ encoding.BinaryAppender    = (*Hasher)(nil)
)

// MarshalBinary returns a stable, versioned encoding of h at its current
// absorption or squeeze position. The encoding includes the customization
// string and can be restored into the zero value of Hasher.
//
// The encoded state is not encrypted or authenticated. It must be protected by
// the caller when either confidentiality or integrity is required.
func (h *Hasher) MarshalBinary() ([]byte, error) {
	return h.AppendBinary(make([]byte, 0, marshaledStateSize+len(h.c)))
}

// AppendBinary appends the stable binary encoding of h to b. It does not
// modify b[:len(b)] or retain b.
func (h *Hasher) AppendBinary(b []byte) ([]byte, error) {
	if err := validateHasherState(h); err != nil {
		return nil, err
	}

	b = append(b, hasherMarshalMagic...)
	b = append(b, hasherMarshalVersion, h.state)
	b = binary.BigEndian.AppendUint64(b, h.pos)
	b = binary.BigEndian.AppendUint64(b, uint64(len(h.c)))
	b = appendMarshaledSponge(b, &h.final)
	b = appendMarshaledSponge(b, &h.leaf)
	b = append(b, h.c...)
	return b, nil
}

// UnmarshalBinary replaces h with the state encoded by data. Data must be a
// complete encoding produced by MarshalBinary or AppendBinary. The receiver is
// unchanged if validation fails, and it does not retain data.
func (h *Hasher) UnmarshalBinary(data []byte) error {
	if len(data) < len(hasherMarshalMagic) || string(data[:len(hasherMarshalMagic)]) != hasherMarshalMagic {
		return errors.New("kt128: invalid hash state identifier")
	}
	if len(data) < marshaledStateSize {
		return errors.New("kt128: invalid hash state size")
	}
	if data[marshalVersionOffset] != hasherMarshalVersion {
		return errors.New("kt128: invalid hash state version")
	}

	customLen := binary.BigEndian.Uint64(data[marshalCustomLenOffset:marshalFinalOffset])
	if customLen != uint64(len(data)-marshaledStateSize) {
		return errors.New("kt128: invalid hash state size")
	}

	decoded := Hasher{
		final: unmarshalSponge(data[marshalFinalOffset:marshalLeafOffset]),
		leaf:  unmarshalSponge(data[marshalLeafOffset:marshaledStateSize]),
		pos:   binary.BigEndian.Uint64(data[marshalPosOffset:marshalCustomLenOffset]),
		state: data[marshalStateOffset],
	}
	switch decoded.state {
	case stateSingle:
		decoded.leafLen = 0
	case stateTree:
		if decoded.pos > ChunkSize {
			decoded.leafLen = int((decoded.pos - ChunkSize) % ChunkSize)
		}
	case stateFinalized:
		decoded.leafLen = 0
	}
	if err := validateHasherState(&decoded); err != nil {
		return err
	}

	custom := append([]byte(nil), data[marshaledStateSize:]...)
	h.c = custom
	h.final = decoded.final
	h.leaf = decoded.leaf
	h.pos = decoded.pos
	h.leafLen = decoded.leafLen
	h.state = decoded.state
	return nil
}

func appendMarshaledSponge(b []byte, s *sponge) []byte {
	for _, lane := range s.a {
		b = binary.LittleEndian.AppendUint64(b, lane)
	}
	return append(b, byte(s.pos))
}

func unmarshalSponge(b []byte) sponge {
	var s sponge
	for i := range s.a {
		s.a[i] = binary.LittleEndian.Uint64(b[i*8:])
	}
	s.pos = int(b[lanes*8])
	return s
}

func validateHasherState(h *Hasher) error {
	switch h.state {
	case stateSingle:
		if h.pos > ChunkSize || h.leafLen != 0 || h.leaf != (sponge{}) {
			return errors.New("kt128: invalid single-node hash state")
		}
		if h.final.pos < 0 || h.final.pos >= rate || h.final.pos != int(h.pos%rate) {
			return errors.New("kt128: invalid single-node sponge position")
		}
		if h.pos < rate && !spongeSuffixIsZero(&h.final, int(h.pos)) {
			return errors.New("kt128: invalid single-node sponge state")
		}

	case stateTree:
		if h.pos <= ChunkSize {
			return errors.New("kt128: invalid tree hash position")
		}
		leafLen := int((h.pos - ChunkSize) % ChunkSize)
		if h.leafLen != leafLen {
			return errors.New("kt128: invalid tree leaf length")
		}
		if h.final.pos < 0 || h.final.pos >= rate || h.final.pos != treeFinalPosition(h.pos) {
			return errors.New("kt128: invalid tree final-node position")
		}
		if h.leaf.pos < 0 || h.leaf.pos >= rate || h.leaf.pos != leafLen%rate {
			return errors.New("kt128: invalid tree leaf position")
		}
		if leafLen == 0 && h.leaf != (sponge{}) {
			return errors.New("kt128: invalid completed tree leaf state")
		}
		if leafLen < rate && !spongeSuffixIsZero(&h.leaf, leafLen) {
			return errors.New("kt128: invalid partial tree leaf state")
		}

	case stateFinalized:
		if h.leafLen != 0 || h.leaf != (sponge{}) {
			return errors.New("kt128: invalid finalized leaf state")
		}
		if h.final.pos < 0 || h.final.pos > rate {
			return errors.New("kt128: invalid finalized sponge position")
		}

	default:
		return errors.New("kt128: invalid hash lifecycle state")
	}
	return nil
}

func treeFinalPosition(pos uint64) int {
	completedLeaves := (pos - ChunkSize) / ChunkSize
	return int((uint64(ChunkSize%rate+len(kt12Marker)) + completedLeaves*32) % rate)
}

// spongeSuffixIsZero validates the untouched portion of a sponge before its
// first permutation, including the capacity lanes beyond the rate.
func spongeSuffixIsZero(s *sponge, pos int) bool {
	lane, rem := pos/8, pos%8
	if rem != 0 {
		if s.a[lane]>>(rem*8) != 0 {
			return false
		}
		lane++
	}
	for ; lane < len(s.a); lane++ {
		if s.a[lane] != 0 {
			return false
		}
	}
	return true
}

package kt128

import (
	"encoding/binary"
	"errors"
)

const (
	hashStateMagic   = "kt128"
	hashStateVersion = 1

	stateKindHash byte = 1
	stateKindXOF  byte = 2

	marshaledSpongeSize = lanes*8 + 1
	marshaledStateSize  = len(hashStateMagic) + 1 + 1 + 1 + 8 + 8 + 8 + 2*marshaledSpongeSize

	marshalVersionOffset   = len(hashStateMagic)
	marshalKindOffset      = marshalVersionOffset + 1
	marshalStateOffset     = marshalKindOffset + 1
	marshalPosOffset       = marshalStateOffset + 1
	marshalCustomLenOffset = marshalPosOffset + 8
	marshalDigestOffset    = marshalCustomLenOffset + 8
	marshalFinalOffset     = marshalDigestOffset + 8
	marshalLeafOffset      = marshalFinalOffset + marshaledSpongeSize
)

func appendState(b []byte, s *state, kind byte, digestSize uint64) ([]byte, error) {
	if err := validateState(s); err != nil {
		return nil, err
	}
	switch kind {
	case stateKindHash:
		if s.phase == stateFinalized || digestSize == 0 {
			return nil, errors.New("kt128: invalid Hash state")
		}
	case stateKindXOF:
		if s.digest != 0 || digestSize != 0 {
			return nil, errors.New("kt128: invalid XOF state")
		}
	default:
		return nil, errors.New("kt128: invalid state kind")
	}

	b = append(b, hashStateMagic...)
	b = append(b, hashStateVersion, kind, s.phase)
	b = binary.BigEndian.AppendUint64(b, s.pos)
	b = binary.BigEndian.AppendUint64(b, uint64(len(s.c)))
	b = binary.BigEndian.AppendUint64(b, digestSize)
	b = appendMarshaledSponge(b, &s.final)
	b = appendMarshaledSponge(b, &s.leaf)
	b = append(b, s.c...)
	return b, nil
}

func unmarshalState(data []byte, wantKind byte) (*state, error) {
	if len(data) < len(hashStateMagic) || string(data[:len(hashStateMagic)]) != hashStateMagic {
		return nil, errors.New("kt128: invalid hash state identifier")
	}
	if len(data) < marshaledStateSize {
		return nil, errors.New("kt128: invalid hash state size")
	}
	if data[marshalVersionOffset] != hashStateVersion {
		return nil, errors.New("kt128: invalid hash state version")
	}
	if data[marshalKindOffset] != wantKind {
		return nil, errors.New("kt128: invalid hash state kind")
	}

	customLen := binary.BigEndian.Uint64(data[marshalCustomLenOffset:marshalDigestOffset])
	if customLen != uint64(len(data)-marshaledStateSize) {
		return nil, errors.New("kt128: invalid hash state size")
	}
	digestSize := binary.BigEndian.Uint64(data[marshalDigestOffset:marshalFinalOffset])
	if wantKind == stateKindHash {
		if digestSize == 0 || digestSize > uint64(^uint(0)>>1) {
			return nil, errors.New("kt128: invalid digest size")
		}
	} else if digestSize != 0 {
		return nil, errors.New("kt128: invalid XOF digest size")
	}

	decoded := &state{
		final:  unmarshalSponge(data[marshalFinalOffset:marshalLeafOffset]),
		leaf:   unmarshalSponge(data[marshalLeafOffset:marshaledStateSize]),
		pos:    binary.BigEndian.Uint64(data[marshalPosOffset:marshalCustomLenOffset]),
		phase:  data[marshalStateOffset],
		digest: int(digestSize),
	}
	switch decoded.phase {
	case stateSingle:
		decoded.leafLen = 0
	case stateTree:
		if decoded.pos > ChunkSize {
			decoded.leafLen = int((decoded.pos - ChunkSize) % ChunkSize)
		}
	case stateFinalized:
		decoded.leafLen = 0
	}
	if wantKind == stateKindHash && decoded.phase == stateFinalized {
		return nil, errors.New("kt128: finalized state cannot be restored as Hash")
	}
	if err := validateState(decoded); err != nil {
		return nil, err
	}
	decoded.c = append([]byte(nil), data[marshaledStateSize:]...)
	return decoded, nil
}

func assignState(dst, src *state) {
	dst.c = src.c
	dst.final = src.final
	dst.leaf = src.leaf
	dst.pos = src.pos
	dst.leafLen = src.leafLen
	dst.phase = src.phase
	dst.digest = src.digest
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

func validateState(h *state) error {
	switch h.phase {
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

package kt128

const (
	// ChunkSize is the KT128 chunk size in bytes.
	ChunkSize = 8192

	// DigestSize is the number of bytes returned by [Hash.Sum].
	DigestSize = 32

	leafDS   = 0x0B
	treeDS   = 0x06
	singleDS = 0x07

	// Internal lifecycle states.
	stateSingle    uint8 = 0 // absorbing, tree mode not entered (<= 1 message chunk)
	stateTree      uint8 = 1 // absorbing, tree mode (S_0 flushed)
	stateFinalized uint8 = 2 // finalized and squeezable
)

// noCopy is recognized by go vet's copylocks analyzer.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}

type state struct {
	noCopy  noCopy
	c       []byte
	final   sponge // final-node sponge state
	leaf    sponge // current partial leaf (tree mode only)
	pos     uint64 // total bytes written via Write
	leafLen int    // bytes absorbed into leaf; 0 = no partial leaf
	phase   uint8  // absorbing or finalized
}

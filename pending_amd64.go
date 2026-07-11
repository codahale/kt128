//go:build amd64 && !purego

package kt128

// pendingState retains an exported partial leaf on architectures whose fused
// S_0 kernels can absorb trailing rate blocks into an otherwise-idle lane.
type pendingState = sponge

func pendingSponge(p *pendingState) *sponge { return p }

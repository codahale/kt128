//go:build !amd64 || purego

package kt128

// pendingState is empty where no fused S_0 kernel can export a partial leaf.
// The generic state machine can only request its sponge after pendingLen has
// been set, which these architectures never do.
type pendingState struct{}

func pendingSponge(*pendingState) *sponge {
	panic("kt128: pending state is unavailable")
}

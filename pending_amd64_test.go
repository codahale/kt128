//go:build amd64 && !purego

package kt128

import (
	"testing"
	"unsafe"
)

func TestAMD64PendingStateRetainsSponge(t *testing.T) {
	if got, want := unsafe.Sizeof(pendingState{}), unsafe.Sizeof(sponge{}); got != want {
		t.Fatalf("pending state size = %d, want sponge size %d", got, want)
	}
	if got, want := unsafe.Sizeof(Hasher{}), uintptr(496); got != want {
		t.Fatalf("Hasher size = %d, want %d", got, want)
	}
}

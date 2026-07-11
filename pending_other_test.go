//go:build !amd64 || purego

package kt128

import (
	"testing"
	"unsafe"
)

func TestPendingStateIsZeroSized(t *testing.T) {
	if got := unsafe.Sizeof(pendingState{}); got != 0 {
		t.Fatalf("pending state size = %d, want 0", got)
	}
	if unsafe.Sizeof(uintptr(0)) == 8 {
		if got, want := unsafe.Sizeof(Hasher{}), uintptr(288); got != want {
			t.Fatalf("Hasher size = %d, want %d", got, want)
		}
	}
}

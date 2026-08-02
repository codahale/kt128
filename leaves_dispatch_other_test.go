//go:build (!amd64 && !arm64) || purego

package kt128

import "testing"

func TestUnavailableKernelWrappers(t *testing.T) {
	testUnavailableKernelWrappers(t)
}

//go:build amd64 && !purego

package kt128

import (
	"unsafe"

	"github.com/codahale/kt128/internal/cpuid"
)

const availableLanes = 8

//go:noescape
func processLeavesAVX512(input *byte, cvs *byte)

//go:noescape
func processLeavesAVX2(input *byte, cvs *byte)

func processLeavesArch(input []byte, cvs *[256]byte) bool {
	if cpuid.HasAVX512 {
		processLeavesAVX512(unsafe.SliceData(input), &cvs[0])
	} else {
		processLeavesAVX2(unsafe.SliceData(input), &cvs[0])
	}
	return true
}

// processLeavesPairArch reports that no 2-wide pair kernel is available on
// amd64; the padded x8 path drains high-utilization remainders instead.
func processLeavesPairArch(_ []byte, _ *[256]byte) bool { return false }

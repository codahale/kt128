//go:build (amd64 || arm64) && !purego && (linux || darwin)

package kt128

import (
	"fmt"
	"runtime/debug"
	"syscall"
	"testing"

	"github.com/codahale/kt128/internal/cpuid"
)

// These tests hedge the memory-safety risk of the leaf kernels, which read
// caller memory directly through unsafe pointers. The remainder kernels are the
// sharpest edge: the AVX-512 run kernel steers its dummy lanes back to chunk 0
// with a clamped gather index, and the AVX2 quad kernel points its dummy lanes
// at an in-bounds chunk. Both are correct only by construction — a wrong clamp
// or offset turns into an out-of-bounds read.
//
// Each kernel is run against a buffer whose final byte sits flush against an
// inaccessible guard page, so any read past the intended end faults. On amd64
// SetPanicOnFault turns the fault into a recoverable panic that names the
// offending kernel; on arm64 the runtime cannot unwind the hand-written kernel
// frame, so the fault aborts the test binary with a fatal error instead. Either
// way a read overrun fails the test loudly.

// byteSink defeats dead-store elimination of the guard-page probe read.
var byteSink byte

// mmapGuarded maps size readable bytes immediately followed by a PROT_NONE guard
// page. It returns a data slice of exactly size bytes ending at the guard
// boundary, the full mapped region, and the offset of the guard page within it.
// The region is unmapped via t.Cleanup. The page size is queried at runtime so
// the layout is correct on hosts with 16 KiB pages (darwin/arm64) as well as
// 4 KiB pages.
func mmapGuarded(t *testing.T, size int) (data, region []byte, guardOff int) {
	t.Helper()

	pageSize := syscall.Getpagesize()
	nDataPages := (size + pageSize - 1) / pageSize
	if nDataPages == 0 {
		nDataPages = 1
	}
	mapLen := nDataPages*pageSize + pageSize

	region, err := syscall.Mmap(-1, 0, mapLen,
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap(%d): %v", mapLen, err)
	}
	t.Cleanup(func() { _ = syscall.Munmap(region) })

	guardOff = nDataPages * pageSize
	if err := syscall.Mprotect(region[guardOff:mapLen], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect: %v", err)
	}

	// Place the data flush against the guard page: data[size-1] is the last
	// readable byte, data[size] is the guard.
	data = region[guardOff-size : guardOff : guardOff]
	return data, region, guardOff
}

// guardedBuffer returns a size-byte slice ending flush against a guard page,
// filled with a non-trivial pattern so the kernels do real reads.
func guardedBuffer(t *testing.T, size int) []byte {
	t.Helper()
	data, _, _ := mmapGuarded(t, size)
	for i := range data {
		data[i] = byte(i*191 + i>>8)
	}
	return data
}

// expectNoFault runs fn and fails the test if it triggers a memory fault.
func expectNoFault(t *testing.T, name string, fn func()) {
	t.Helper()
	old := debug.SetPanicOnFault(true)
	defer func() {
		debug.SetPanicOnFault(old)
		if r := recover(); r != nil {
			t.Errorf("%s read past the guard page: %v", name, r)
		}
	}()
	fn()
}

// expectFault runs fn and fails the test unless it triggers a memory fault. It
// validates that the guard page is actually armed and that faults are caught.
func expectFault(t *testing.T, name string, fn func()) {
	t.Helper()
	old := debug.SetPanicOnFault(true)
	faulted := false
	func() {
		defer func() {
			if recover() != nil {
				faulted = true
			}
		}()
		fn()
	}()
	debug.SetPanicOnFault(old)
	if !faulted {
		t.Errorf("%s: expected a guard-page fault, got none", name)
	}
}

// TestGuardPageArmed is a negative control: it confirms the mmap/mprotect/recover
// machinery actually faults on a read one byte past the data region. Without
// this, a positive no-fault result could mean the guard was never armed.
func TestGuardPageArmed(t *testing.T) {
	_, region, guardOff := mmapGuarded(t, 8)
	byteSink = region[guardOff-1] // last readable byte: must not fault
	expectFault(t, "guard page", func() {
		byteSink = region[guardOff] // first guard byte: must fault
	})
}

// TestLeafKernelsNoReadOverrun runs every available leaf kernel against
// guard-flush buffers. On an AVX-512 host it also forces the AVX2/quad path so
// both kernel families are exercised on the same machine.
func TestLeafKernelsNoReadOverrun(t *testing.T) {
	t.Run("native", func(t *testing.T) { runLeafKernels(t) })

	if cpuid.HasAVX512 {
		defer func() { cpuid.HasAVX512 = true }()
		cpuid.HasAVX512 = false
		t.Run("avx2", func(t *testing.T) { runLeafKernels(t) })
	}
}

func runLeafKernels(t *testing.T) {
	t.Helper()
	var cvs [256]byte

	// x8 fused kernel: reads exactly 8 contiguous chunks.
	if hasLeafX8 {
		expectNoFault(t, "processLeavesArch(x8)", func() {
			processLeavesArch(guardedBuffer(t, 8*BlockSize), &cvs)
		})
	}

	// Probe kernel availability with throwaway heap buffers so the guarded run
	// below is a single, clean call.
	var probe [256]byte
	hasPair := processLeavesPairArch(make([]byte, 2*BlockSize), &probe)
	hasRun := processLeavesRunArch(make([]byte, 2*BlockSize), 2, &probe)
	hasBatch5 := processLeavesBatch5Arch(make([]byte, 5*BlockSize), &probe)
	hasTriple := processLeavesTripleArch(make([]byte, 3*BlockSize), &probe)

	// x5 hybrid kernel (arm64): the scalar walker must end exactly at the
	// buffer end and the NEON walkers within it.
	if hasBatch5 {
		expectNoFault(t, "processLeavesBatch5(x5)", func() {
			processLeavesBatch5Arch(guardedBuffer(t, 5*BlockSize), &cvs)
		})
	}

	if hasTriple {
		expectNoFault(t, "processLeavesTriple(x3)", func() {
			processLeavesTripleArch(guardedBuffer(t, 3*BlockSize), &cvs)
		})
	}

	// x2 pair kernel (arm64): reads exactly 2 contiguous chunks.
	if hasPair {
		expectNoFault(t, "processLeavesPair(x2)", func() {
			processLeavesPairArch(guardedBuffer(t, 2*BlockSize), &cvs)
		})
	}

	// Fused S_0+leaves kernel: must read exactly n contiguous chunks. On
	// AVX-512 the dummy lanes must stay clamped within chunk 0.
	for n := 2; n <= availableLanes; n++ {
		var probeFinal sponge
		var probeCVs [256]byte
		if !processS0LeavesArch(make([]byte, n*BlockSize), n, &probeFinal, &probeCVs) {
			continue
		}
		buf := guardedBuffer(t, n*BlockSize)
		var final sponge
		var cvs [256]byte
		expectNoFault(t, fmt.Sprintf("processS0Leaves(n=%d)", n), func() {
			processS0LeavesArch(buf, n, &final, &cvs)
		})
	}

	// Fused S_0+leaves+partial kernel: reads exactly n contiguous chunks plus
	// nShared whole rate-blocks of the trailing partial. The tail lane is
	// re-clamped to a dummy after its blocks; a wrong clamp or an unclamped
	// walk past the partial's blocks reads the guard.
	for n := 2; n <= 7; n++ {
		var probeFinal, probePending sponge
		var probeCVs [256]byte
		if !processS0LeavesTailArch(make([]byte, n*BlockSize), n, 0, &probeFinal, &probePending, &probeCVs) {
			continue
		}
		for _, nShared := range []int{0, 24, 48} {
			buf := guardedBuffer(t, n*BlockSize+nShared*rate)
			var final, pending sponge
			var cvsOut [256]byte
			expectNoFault(t, fmt.Sprintf("processS0LeavesTail(n=%d,nShared=%d)", n, nShared), func() {
				processS0LeavesTailArch(buf, n, nShared, &final, &pending, &cvsOut)
			})
		}
	}

	// Run/quad remainder kernels (amd64): the dummy lanes must stay within the
	// n-chunk buffer. A clamping bug would read chunk n (past the guard).
	if hasRun {
		for n := 2; n <= 7; n++ {
			buf := guardedBuffer(t, n*BlockSize)
			expectNoFault(t, fmt.Sprintf("processLeavesRun(n=%d)", n), func() {
				processLeavesRunArch(buf, n, &cvs)
			})
		}
	}

	// Trailing-leaves+partial kernel: reads exactly n*BlockSize complete-chunk
	// bytes plus nShared whole rate-blocks of the head. On AVX-512 the tail
	// lane is re-clamped to a dummy after its blocks; a wrong clamp or an
	// unclamped walk past the head reads the guard.
	for n := 1; n <= 7; n++ {
		var probeCVs [256]byte
		var probeSponge sponge
		if !processLeavesTailArch(make([]byte, n*BlockSize), n, 0, &probeCVs, &probeSponge) {
			continue
		}
		const headLen = 25 * rate
		buf := guardedBuffer(t, n*BlockSize+headLen)
		var cvsOut [256]byte
		var s sponge
		expectNoFault(t, fmt.Sprintf("processLeavesTail(n=%d)", n), func() {
			processLeavesTailArch(buf, n, headLen/rate, &cvsOut, &s)
		})
	}

	// Scalar fused absorb loop: reads exactly n bytes from its pointer.
	const stripes = 48
	var s sponge
	buf := guardedBuffer(t, stripes*rate)
	expectNoFault(t, "fastLoopAbsorb168x1", func() {
		fastLoopAbsorb168x1(&s, &buf[0], stripes*rate)
	})
}

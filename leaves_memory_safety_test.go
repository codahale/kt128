//go:build (amd64 || arm64) && !purego && (linux || darwin)

package kt128

import (
	"fmt"
	"runtime/debug"
	"syscall"
	"testing"
	"unsafe"

	"github.com/codahale/kt128/internal/cpuid"
)

// These tests hedge the memory-safety risk of the leaf kernels, which read
// caller memory directly through unsafe pointers. The remainder kernels are the
// sharpest edge: inactive AVX-512 lanes must remain masked, and the AVX2 quad
// kernel points its dummy lanes at an in-bounds chunk. Both are correct only by
// construction — a wrong mask, pointer, or offset turns into an out-of-bounds
// read.
//
// Each kernel is run twice against read-only input: once flush against a leading
// guard page and once flush against a trailing guard page. Underreads, overreads,
// and accidental input writes therefore fault. Writable outputs are likewise
// placed against both guard edges in TestAssemblyOutputBounds. On amd64
// SetPanicOnFault turns a fault into a recoverable panic that names the offending
// kernel; on arm64 the runtime cannot unwind the hand-written kernel frame, so
// the fault aborts the test binary with a fatal error instead. Either way an
// out-of-contract memory access fails the test loudly.

// byteSink defeats dead-store elimination of the guard-page probe read.
var byteSink byte

type guardEdge uint8

const (
	guardBefore guardEdge = iota
	guardAfter
)

func (e guardEdge) String() string {
	if e == guardBefore {
		return "before"
	}
	return "after"
}

var guardEdges = [...]guardEdge{guardBefore, guardAfter}

type assemblySafetyPermuter struct {
	name string
	fn   func(*sponge)
}

type assemblySafetyAbsorber struct {
	name string
	fn   func(*sponge, *byte, int)
}

// mmapGuardedAt maps a writable page-aligned region between two PROT_NONE guard
// pages and returns an exact-capacity size-byte view flush against edge. The
// complete mapping is retained for fault probes and captured by t.Cleanup.
func mmapGuardedAt(t *testing.T, size int, edge guardEdge) (data, region, pages []byte, afterOff int) {
	t.Helper()
	if size <= 0 {
		t.Fatalf("guarded mapping size must be positive: %d", size)
	}

	pageSize := syscall.Getpagesize()
	nDataPages := (size + pageSize - 1) / pageSize
	dataLen := nDataPages * pageSize
	mapLen := dataLen + 2*pageSize

	region, err := syscall.Mmap(-1, 0, mapLen,
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap(%d): %v", mapLen, err)
	}
	t.Cleanup(func() { _ = syscall.Munmap(region) })

	afterOff = pageSize + dataLen
	if err := syscall.Mprotect(region[:pageSize], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect leading guard: %v", err)
	}
	if err := syscall.Mprotect(region[afterOff:], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect trailing guard: %v", err)
	}

	pages = region[pageSize:afterOff]
	start := 0
	if edge == guardAfter {
		start = len(pages) - size
	}
	data = pages[start : start+size : start+size]
	return data, region, pages, afterOff
}

// mmapGuarded maps size readable bytes immediately followed by a PROT_NONE guard
// page. It returns a data slice of exactly size bytes ending at the guard
// boundary, the full mapped region, and the offset of the guard page within it.
// The region is unmapped via t.Cleanup. The page size is queried at runtime so
// the layout is correct on hosts with 16 KiB pages (darwin/arm64) as well as
// 4 KiB pages.
func mmapGuarded(t *testing.T, size int) (data, region []byte, guardOff int) {
	t.Helper()
	data, region, _, guardOff = mmapGuardedAt(t, size, guardAfter)
	return data, region, guardOff
}

// guardedInput returns patterned input flush against edge and marks all of its
// backing pages read-only, so an assembly write faults as well as an out-of-bounds
// read. Padding between data and the opposite guard is inaccessible to Go callers
// through the exact-capacity returned slice but remains readable by assembly; the
// second edge run closes that side of the contract.
func guardedInput(t *testing.T, size int, edge guardEdge) []byte {
	t.Helper()
	data, _, pages, _ := mmapGuardedAt(t, size, edge)
	for i := range data {
		data[i] = byte(i*191 + i>>8)
	}
	if err := syscall.Mprotect(pages, syscall.PROT_READ); err != nil {
		t.Fatalf("mprotect input read-only: %v", err)
	}
	return data
}

func checkGuardedInput(t *testing.T, name string, size int, fn func([]byte)) {
	t.Helper()
	for _, edge := range guardEdges {
		buf := guardedInput(t, size, edge)
		expectNoFault(t, name+"/guard-"+edge.String(), func() { fn(buf) })
	}
}

// guardedValue returns a zeroed T flush against edge. The safety tests use it
// only for pointer-free byte arrays and sponge state.
func guardedValue[T any](t *testing.T, edge guardEdge) *T {
	t.Helper()
	var zero T
	data, _, _, _ := mmapGuardedAt(t, int(unsafe.Sizeof(zero)), edge)
	p := unsafe.Pointer(unsafe.SliceData(data))
	if uintptr(p)%unsafe.Alignof(zero) != 0 {
		t.Fatalf("guarded %T is not aligned", zero)
	}
	return (*T)(p)
}

// expectNoFault runs fn and fails the test if it triggers a memory fault.
func expectNoFault(t *testing.T, name string, fn func()) {
	t.Helper()
	old := debug.SetPanicOnFault(true)
	defer func() {
		debug.SetPanicOnFault(old)
		if r := recover(); r != nil {
			t.Errorf("%s violated guarded memory: %v", name, r)
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
	expectFault(t, "trailing guard page", func() {
		byteSink = region[guardOff] // first guard byte: must fault
	})
	pageSize := syscall.Getpagesize()
	expectFault(t, "leading guard page", func() {
		byteSink = region[pageSize-1] // final byte of the leading guard
	})
}

// TestAssemblyInputReadOnly is the negative control for guardedInput: writes to
// its backing pages must fault before the assembly tests rely on that protection.
func TestAssemblyInputReadOnly(t *testing.T) {
	buf := guardedInput(t, 8, guardBefore)
	expectFault(t, "read-only input", func() {
		buf[0] = 1
	})
}

// TestAssemblyInputBounds runs every available leaf kernel against read-only
// input flush with both guard edges. On an AVX-512 host it also forces the
// AVX2/quad path so both kernel families are exercised on the same machine.
func TestAssemblyInputBounds(t *testing.T) {
	t.Run("native", func(t *testing.T) { runLeafKernels(t) })

	if cpuid.HasAVX512 && cpuid.HasAVX2 {
		hadAVX512 := cpuid.HasAVX512
		defer func() { cpuid.HasAVX512 = hadAVX512 }()
		cpuid.HasAVX512 = false
		t.Run("avx2", func(t *testing.T) { runLeafKernels(t) })
	}
}

const outputCanary = 0xA5

// checkCVOutput guards the complete scratch array and, when written >= 0,
// verifies that bytes beyond the kernel's documented output prefix are intact.
// Some AVX2 wrappers intentionally store dummy-lane CVs which callers ignore;
// those use written == -1 and are constrained only to the full scratch object.
func checkCVOutput(t *testing.T, name string, written int, fn func(*[256]byte)) {
	t.Helper()
	for _, edge := range guardEdges {
		cvs := guardedValue[[256]byte](t, edge)
		for i := range cvs {
			cvs[i] = outputCanary
		}
		expectNoFault(t, name+"/guard-"+edge.String(), func() { fn(cvs) })
		if written < 0 {
			continue
		}
		for i, b := range cvs[written:] {
			if b != outputCanary {
				t.Fatalf("%s wrote cvs[%d] outside documented prefix %d", name, written+i, written)
			}
		}
	}
}

func checkSpongeOutput(t *testing.T, name string, fn func(*sponge)) {
	t.Helper()
	for _, edge := range guardEdges {
		s := guardedValue[sponge](t, edge)
		expectNoFault(t, name+"/guard-"+edge.String(), func() { fn(s) })
	}
}

// TestAssemblyOutputBounds guards every CV and sponge output accepted by the
// architecture wrappers. Prefix canaries additionally pin the narrower output
// footprints promised by the pair, triple, and batch-five wrappers.
func TestAssemblyOutputBounds(t *testing.T) {
	t.Run("native", func(t *testing.T) { runLeafKernelOutputBounds(t) })

	if cpuid.HasAVX512 && cpuid.HasAVX2 {
		hadAVX512 := cpuid.HasAVX512
		defer func() { cpuid.HasAVX512 = hadAVX512 }()
		cpuid.HasAVX512 = false
		t.Run("avx2", func(t *testing.T) { runLeafKernelOutputBounds(t) })
	}
}

func runLeafKernelOutputBounds(t *testing.T) {
	t.Helper()

	if hasLeafX8 {
		input := make([]byte, 8*ChunkSize)
		checkCVOutput(t, "processLeavesArch(x8)", 256, func(cvs *[256]byte) {
			processLeavesArch(input, cvs)
		})
	}

	var probe [256]byte
	hasPair := processLeavesPairArch(make([]byte, 2*ChunkSize), &probe)
	hasRun := processLeavesRunArch(make([]byte, 2*ChunkSize), 2, &probe)
	hasBatch5 := processLeavesBatch5Arch(make([]byte, 5*ChunkSize), &probe)
	hasTriple := processLeavesTripleArch(make([]byte, 3*ChunkSize), &probe)

	if hasBatch5 {
		input := make([]byte, 5*ChunkSize)
		checkCVOutput(t, "processLeavesBatch5(x5)", 5*32, func(cvs *[256]byte) {
			processLeavesBatch5Arch(input, cvs)
		})
	}
	if hasTriple {
		input := make([]byte, 3*ChunkSize)
		checkCVOutput(t, "processLeavesTriple(x3)", 3*32, func(cvs *[256]byte) {
			processLeavesTripleArch(input, cvs)
		})
	}
	if hasPair {
		input := make([]byte, 2*ChunkSize)
		checkCVOutput(t, "processLeavesPair(x2)", 2*32, func(cvs *[256]byte) {
			processLeavesPairArch(input, cvs)
		})
	}
	if hasRun {
		for n := 2; n <= 7; n++ {
			input := make([]byte, n*ChunkSize)
			checkCVOutput(t, fmt.Sprintf("processLeavesRun(n=%d)", n), -1, func(cvs *[256]byte) {
				processLeavesRunArch(input, n, cvs)
			})
		}
	}

	for n := 2; n <= availableLanes; n++ {
		input := make([]byte, n*ChunkSize)
		var probeFinal sponge
		var probeCVs [256]byte
		if !processS0LeavesArch(input, n, &probeFinal, &probeCVs) {
			continue
		}
		name := fmt.Sprintf("processS0Leaves(n=%d)", n)
		checkCVOutput(t, name+"/cvs", -1, func(cvs *[256]byte) {
			var final sponge
			processS0LeavesArch(input, n, &final, cvs)
		})
		checkSpongeOutput(t, name+"/final", func(final *sponge) {
			var cvs [256]byte
			processS0LeavesArch(input, n, final, &cvs)
		})
	}

	for n := 2; n <= 7; n++ {
		for _, nShared := range []int{0, 1, 48} {
			input := make([]byte, n*ChunkSize+nShared*rate)
			var probeFinal, probePending sponge
			var probeCVs [256]byte
			if !processS0LeavesTailArch(input, n, nShared, &probeFinal, &probePending, &probeCVs) {
				continue
			}
			name := fmt.Sprintf("processS0LeavesTail(n=%d,nShared=%d)", n, nShared)
			checkCVOutput(t, name+"/cvs", -1, func(cvs *[256]byte) {
				var final, pending sponge
				processS0LeavesTailArch(input, n, nShared, &final, &pending, cvs)
			})
			checkSpongeOutput(t, name+"/final", func(final *sponge) {
				var pending sponge
				var cvs [256]byte
				processS0LeavesTailArch(input, n, nShared, final, &pending, &cvs)
			})
			checkSpongeOutput(t, name+"/pending", func(pending *sponge) {
				var final sponge
				var cvs [256]byte
				processS0LeavesTailArch(input, n, nShared, &final, pending, &cvs)
			})
		}
	}

	for n := 1; n <= 7; n++ {
		for _, nShared := range []int{0, 1, 48} {
			input := make([]byte, n*ChunkSize+nShared*rate)
			var probeCVs [256]byte
			var probePartial sponge
			if !processLeavesTailArch(input, n, nShared, &probeCVs, &probePartial) {
				continue
			}
			name := fmt.Sprintf("processLeavesTail(n=%d,nShared=%d)", n, nShared)
			checkCVOutput(t, name+"/cvs", -1, func(cvs *[256]byte) {
				var partial sponge
				processLeavesTailArch(input, n, nShared, cvs, &partial)
			})
			checkSpongeOutput(t, name+"/partial", func(partial *sponge) {
				var cvs [256]byte
				processLeavesTailArch(input, n, nShared, &cvs, partial)
			})
		}
	}
}

// TestAssemblyPrimitiveStateBounds places the in-place Keccak state against
// both guard edges for every available permutation and fused absorb primitive.
// Correctness remains checked independently by the corresponding fuzz targets;
// the sentinel pins the assembly write footprint to sponge.a.
func TestAssemblyPrimitiveStateBounds(t *testing.T) {
	for _, permute := range assemblySafetyPermuters() {
		checkSpongeOutput(t, permute.name, func(s *sponge) {
			s.pos = posSentinel
			permute.fn(s)
			if s.pos != posSentinel {
				t.Fatalf("%s modified sponge.pos: got %#x, want %#x", permute.name, s.pos, posSentinel)
			}
		})
	}

	input := splitmixBytes(48*rate, 0xA55E)
	for _, absorb := range assemblySafetyAbsorbers() {
		checkSpongeOutput(t, absorb.name, func(s *sponge) {
			s.pos = posSentinel
			absorb.fn(s, &input[0], len(input))
			if s.pos != posSentinel {
				t.Fatalf("%s modified sponge.pos: got %#x, want %#x", absorb.name, s.pos, posSentinel)
			}
		})
	}
}

func runLeafKernels(t *testing.T) {
	t.Helper()
	var cvs [256]byte

	// x8 fused kernel: reads exactly 8 contiguous chunks.
	if hasLeafX8 {
		checkGuardedInput(t, "processLeavesArch(x8)", 8*ChunkSize, func(buf []byte) {
			processLeavesArch(buf, &cvs)
		})
	}

	// Probe kernel availability with throwaway heap buffers so the guarded run
	// below is a single, clean call.
	var probe [256]byte
	hasPair := processLeavesPairArch(make([]byte, 2*ChunkSize), &probe)
	hasRun := processLeavesRunArch(make([]byte, 2*ChunkSize), 2, &probe)
	hasBatch5 := processLeavesBatch5Arch(make([]byte, 5*ChunkSize), &probe)
	hasTriple := processLeavesTripleArch(make([]byte, 3*ChunkSize), &probe)

	// x5 hybrid kernel (arm64): the scalar walker must end exactly at the
	// buffer end and the NEON walkers within it.
	if hasBatch5 {
		checkGuardedInput(t, "processLeavesBatch5(x5)", 5*ChunkSize, func(buf []byte) {
			processLeavesBatch5Arch(buf, &cvs)
		})
	}

	if hasTriple {
		checkGuardedInput(t, "processLeavesTriple(x3)", 3*ChunkSize, func(buf []byte) {
			processLeavesTripleArch(buf, &cvs)
		})
	}

	// x2 pair kernel (arm64): reads exactly 2 contiguous chunks.
	if hasPair {
		checkGuardedInput(t, "processLeavesPair(x2)", 2*ChunkSize, func(buf []byte) {
			processLeavesPairArch(buf, &cvs)
		})
	}

	// Fused S_0+leaves kernel: must read exactly n contiguous chunks. On
	// AVX-512 the dummy lanes must stay clamped within chunk 0.
	for n := 2; n <= availableLanes; n++ {
		var probeFinal sponge
		var probeCVs [256]byte
		if !processS0LeavesArch(make([]byte, n*ChunkSize), n, &probeFinal, &probeCVs) {
			continue
		}
		checkGuardedInput(t, fmt.Sprintf("processS0Leaves(n=%d)", n), n*ChunkSize, func(buf []byte) {
			var final sponge
			var cvs [256]byte
			processS0LeavesArch(buf, n, &final, &cvs)
		})
	}

	// Fused S_0+leaves+partial kernel: exhaustively test every supported lane
	// count and shared-stripe count. It reads exactly n contiguous chunks plus
	// nShared whole rate-blocks of the trailing partial. After those blocks the
	// tail lane must be masked off; a wrong mask or an overlong walk reads the
	// guard.
	for n := 2; n <= 7; n++ {
		var probeFinal, probePending sponge
		var probeCVs [256]byte
		if !processS0LeavesTailArch(make([]byte, n*ChunkSize), n, 0, &probeFinal, &probePending, &probeCVs) {
			continue
		}
		for nShared := 0; nShared <= 48; nShared++ {
			checkGuardedInput(t, fmt.Sprintf("processS0LeavesTail(n=%d,nShared=%d)", n, nShared), n*ChunkSize+nShared*rate, func(buf []byte) {
				var final, pending sponge
				var cvsOut [256]byte
				processS0LeavesTailArch(buf, n, nShared, &final, &pending, &cvsOut)
			})
		}
	}

	// Run/quad remainder kernels (amd64): the dummy lanes must stay within the
	// n-chunk buffer. A clamping bug would read chunk n (past the guard).
	if hasRun {
		for n := 2; n <= 7; n++ {
			checkGuardedInput(t, fmt.Sprintf("processLeavesRun(n=%d)", n), n*ChunkSize, func(buf []byte) {
				processLeavesRunArch(buf, n, &cvs)
			})
		}
	}

	// Trailing-leaves+partial kernel: exhaustively test every supported lane
	// count and shared-stripe count. It reads exactly n*ChunkSize complete-chunk
	// bytes plus nShared whole rate-blocks of the head. On AVX-512 the tail lane
	// must be masked off after its blocks; a wrong mask or an overlong walk reads
	// the guard.
	for n := 1; n <= 7; n++ {
		var probeCVs [256]byte
		var probeSponge sponge
		if !processLeavesTailArch(make([]byte, n*ChunkSize), n, 0, &probeCVs, &probeSponge) {
			continue
		}
		for nShared := 0; nShared <= 48; nShared++ {
			checkGuardedInput(t, fmt.Sprintf("processLeavesTail(n=%d,nShared=%d)", n, nShared), n*ChunkSize+nShared*rate, func(buf []byte) {
				var cvsOut [256]byte
				var s sponge
				processLeavesTailArch(buf, n, nShared, &cvsOut, &s)
			})
		}
	}

	// Scalar and ISA-specific fused absorb loops read exactly n bytes from their
	// input pointer and leave sponge.pos untouched.
	const stripes = 48
	for _, absorb := range assemblySafetyAbsorbers() {
		checkGuardedInput(t, absorb.name, stripes*rate, func(buf []byte) {
			var s sponge
			s.pos = posSentinel
			absorb.fn(&s, &buf[0], len(buf))
			if s.pos != posSentinel {
				t.Fatalf("%s modified sponge.pos: got %#x, want %#x", absorb.name, s.pos, posSentinel)
			}
		})
	}
}

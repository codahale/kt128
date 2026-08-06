//go:build (amd64 || arm64) && !purego

package kt128

import "testing"

// TestKernelWrapperBoundsAssertions verifies that each kernel wrapper panics,
// before any assembly runs, on an input that does not cover its documented
// read footprint, and that the tail wrappers reject an nShared beyond the
// kernels' stripe bound. Each case first probes the shape with a
// contract-valid call; shapes with no kernel on this CPU return false there
// and are skipped.
func TestKernelWrapperBoundsAssertions(t *testing.T) {
	cases := []struct {
		name  string
		probe func() bool
		bad   func()
	}{
		{
			name: "x8 short input",
			probe: func() bool {
				var cvs [256]byte
				return tryProcessLeavesX8Arch(make([]byte, 8*ChunkSize), &cvs)
			},
			bad: func() {
				var cvs [256]byte
				tryProcessLeavesX8Arch(make([]byte, 8*ChunkSize-1), &cvs)
			},
		},
		{
			name: "batch5 short input",
			probe: func() bool {
				var cvs [256]byte
				return tryProcessLeavesBatch5Arch(make([]byte, 5*ChunkSize), &cvs)
			},
			bad: func() {
				var cvs [256]byte
				tryProcessLeavesBatch5Arch(make([]byte, 5*ChunkSize-1), &cvs)
			},
		},
		{
			name: "triple short input",
			probe: func() bool {
				var cvs [256]byte
				return tryProcessLeavesTripleArch(make([]byte, 3*ChunkSize), &cvs)
			},
			bad: func() {
				var cvs [256]byte
				tryProcessLeavesTripleArch(make([]byte, 3*ChunkSize-1), &cvs)
			},
		},
		{
			name: "pair short input",
			probe: func() bool {
				var cvs [256]byte
				return tryProcessLeavesPairArch(make([]byte, 2*ChunkSize), &cvs)
			},
			bad: func() {
				var cvs [256]byte
				tryProcessLeavesPairArch(make([]byte, 2*ChunkSize-1), &cvs)
			},
		},
		{
			name: "run short input",
			probe: func() bool {
				var cvs [256]byte
				return tryProcessLeavesRunArch(make([]byte, 3*ChunkSize), 3, &cvs)
			},
			bad: func() {
				var cvs [256]byte
				tryProcessLeavesRunArch(make([]byte, 3*ChunkSize-1), 3, &cvs)
			},
		},
		{
			name: "s0 leaves pair short input",
			probe: func() bool {
				var final sponge
				var cvs [256]byte
				return tryProcessS0LeavesArch(make([]byte, 2*ChunkSize), 2, &final, &cvs)
			},
			bad: func() {
				var final sponge
				var cvs [256]byte
				tryProcessS0LeavesArch(make([]byte, 2*ChunkSize-1), 2, &final, &cvs)
			},
		},
		{
			name: "s0 leaves triple short input",
			probe: func() bool {
				var final sponge
				var cvs [256]byte
				return tryProcessS0LeavesArch(make([]byte, 3*ChunkSize), 3, &final, &cvs)
			},
			bad: func() {
				var final sponge
				var cvs [256]byte
				tryProcessS0LeavesArch(make([]byte, 3*ChunkSize-1), 3, &final, &cvs)
			},
		},
		{
			name: "s0 tail short input",
			probe: func() bool {
				var final, pending sponge
				var cvs [256]byte
				return tryProcessS0LeavesTailArch(make([]byte, 2*ChunkSize+rate), 2, 1, &final, &pending, &cvs)
			},
			bad: func() {
				var final, pending sponge
				var cvs [256]byte
				tryProcessS0LeavesTailArch(make([]byte, 2*ChunkSize+rate-1), 2, 1, &final, &pending, &cvs)
			},
		},
		{
			name: "s0 tail stripe bound",
			probe: func() bool {
				var final, pending sponge
				var cvs [256]byte
				return tryProcessS0LeavesTailArch(make([]byte, 2*ChunkSize+rate), 2, 1, &final, &pending, &cvs)
			},
			bad: func() {
				var final, pending sponge
				var cvs [256]byte
				tryProcessS0LeavesTailArch(make([]byte, 2*ChunkSize+49*rate), 2, 49, &final, &pending, &cvs)
			},
		},
		{
			name: "leaves tail short input",
			probe: func() bool {
				var partial sponge
				var cvs [256]byte
				return tryProcessLeavesTailArch(make([]byte, ChunkSize+rate), 1, 1, &cvs, &partial)
			},
			bad: func() {
				var partial sponge
				var cvs [256]byte
				tryProcessLeavesTailArch(make([]byte, ChunkSize+rate-1), 1, 1, &cvs, &partial)
			},
		},
		{
			name: "leaves tail stripe bound",
			probe: func() bool {
				var partial sponge
				var cvs [256]byte
				return tryProcessLeavesTailArch(make([]byte, ChunkSize+rate), 1, 1, &cvs, &partial)
			},
			bad: func() {
				var partial sponge
				var cvs [256]byte
				tryProcessLeavesTailArch(make([]byte, ChunkSize+49*rate), 1, 49, &cvs, &partial)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.probe() {
				t.Skip("no kernel for this shape on this CPU")
			}
			defer func() {
				if recover() == nil {
					t.Error("expected bounds panic, got none")
				}
			}()
			tc.bad()
		})
	}
}

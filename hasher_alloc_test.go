package kt128

import "testing"

// TestWriteAllocationsDoNotScale guards the zero-copy Write/finalize paths. The
// large-write fast path processes leaves directly from the caller's buffer and
// finalization never reallocates it, so a full New/Write/Read cycle allocates a
// small constant number of times regardless of message size. A regression that
// copied or buffered proportionally to the input would make the larger message
// allocate more. Both sizes are well past lanes*BlockSize, so both take the
// fast path.
func TestWriteAllocationsDoNotScale(t *testing.T) {
	out := make([]byte, 32)
	cycle := func(msg []byte) func() {
		return func() {
			h := New(nil)
			_, _ = h.Write(msg)
			_, _ = h.Read(out)
		}
	}

	med := ptn(128 * 1024)
	big := ptn(1024 * 1024)

	aMed := testing.AllocsPerRun(10, cycle(med))
	aBig := testing.AllocsPerRun(10, cycle(big))
	t.Logf("allocs/cycle: 128KiB=%.0f 1MiB=%.0f", aMed, aBig)

	if aBig > aMed {
		t.Errorf("Write/finalize allocations scale with input: %.0f at 128 KiB but %.0f at 1 MiB", aMed, aBig)
	}
	if aMed > 8 {
		t.Errorf("New/Write/Read allocated %.0f times, want a small constant", aMed)
	}
}

// TestReadAllocationsDoNotScale guards that squeeze writes output directly into
// the caller's buffer: producing more output must not allocate more. The output
// buffers are allocated outside the measured cycle so only squeeze's own
// allocations are counted.
func TestReadAllocationsDoNotScale(t *testing.T) {
	msg := ptn(BlockSize + 1) // tree mode
	out32 := make([]byte, 32)
	out4k := make([]byte, 4096)

	read := func(out []byte) func() {
		return func() {
			h := New(nil)
			_, _ = h.Write(msg)
			_, _ = h.Read(out)
		}
	}

	a32 := testing.AllocsPerRun(20, read(out32))
	a4k := testing.AllocsPerRun(20, read(out4k))
	t.Logf("allocs/cycle: out=32B=%.0f out=4096B=%.0f", a32, a4k)

	if a4k > a32 {
		t.Errorf("output allocations scale with length: %.0f for 32 B but %.0f for 4096 B", a32, a4k)
	}
}

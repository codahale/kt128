package kt128

import (
	"bytes"
	"fmt"
	"sync"
	"testing"
)

// These tests drive independent hashers, clones, and read-only operations from
// many goroutines at once. Run under -race (the standard CI configs do), they
// assert that the package holds no shared mutable state: clones evolve
// independently, the SIMD leaf kernels are reentrant, and Clone/Equal/Pos never
// mutate their receiver. The race detector instruments the Go-level buffering and
// state machine; the assembly kernels take only caller-owned pointers (input,
// cvs, sponge) and touch no package globals, so they are reentrant by
// construction and the correctness checks here exercise them concurrently. Each
// goroutine reports through its own result slot and never calls t.Fatal, which
// must not be used outside the test goroutine.

// TestConcurrentCloneIndependence clones one tree-mode hasher into many
// goroutines, evolves each clone with distinct data, and confirms every clone
// matches a sequentially computed reference and that the shared source is
// unchanged. Clone reads the source (and its shared customization slice) from
// every goroutine at once, so a race here means Clone is not a pure read.
func TestConcurrentCloneIndependence(t *testing.T) {
	const workers = 8
	custom := []byte("ctx")
	baseMsg := ptn(2*BlockSize + 100) // tree mode: Clone copies buf + sponge + leafCount

	base := New(custom)
	if _, err := base.Write(baseMsg); err != nil {
		t.Fatalf("Write: %v", err)
	}

	extra := make([][]byte, workers)
	for w := range extra {
		extra[w] = lcFill(byte(w+1), 137*(w+1)) // distinct per worker
	}

	got := make([][]byte, workers)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			h := base.Clone()
			_, _ = h.Write(extra[w])
			out := make([]byte, 64)
			_, _ = h.Read(out)
			got[w] = out
		}(w)
	}
	wg.Wait()

	for w := range workers {
		fullMsg := append(append([]byte{}, baseMsg...), extra[w]...)
		want := referenceKT128(fullMsg, custom, 64)
		if !bytes.Equal(got[w], want) {
			t.Errorf("worker %d: clone output mismatch\n got  %x\n want %x", w, got[w], want)
		}
	}

	// The source must be untouched by the concurrent clones: reading a fresh clone
	// of it (so base itself stays unfinalized) still yields the base output.
	baseOut := make([]byte, 64)
	_, _ = base.Clone().Read(baseOut)
	if want := referenceKT128(baseMsg, custom, 64); !bytes.Equal(baseOut, want) {
		t.Errorf("source hasher changed after concurrent clones\n got  %x\n want %x", baseOut, want)
	}
}

// TestConcurrentIndependentHashers runs many fully independent New/Write/Read
// cycles concurrently across sizes and customizations that span the single-node,
// tree, and multi-leaf SIMD paths, confirming each result against referenceKT128.
// This is the broad reentrancy stress on the leaf kernels and the absorb loop.
func TestConcurrentIndependentHashers(t *testing.T) {
	sizes := []int{0, 1, BlockSize - 1, BlockSize, BlockSize + 1, 2*BlockSize + 50, 9 * BlockSize}
	customs := [][]byte{nil, []byte("d"), ptn(41)}
	const iters = 4

	type task struct {
		size   int
		custom []byte
	}
	var tasks []task
	for _, sz := range sizes {
		for _, cu := range customs {
			for range iters {
				tasks = append(tasks, task{sz, cu})
			}
		}
	}

	errs := make([]error, len(tasks))
	var wg sync.WaitGroup
	for i, tk := range tasks {
		wg.Add(1)
		go func(i int, tk task) {
			defer wg.Done()
			msg := ptn(tk.size)
			h := New(tk.custom)
			_, _ = h.Write(msg)
			out := make([]byte, 100)
			_, _ = h.Read(out)
			if want := referenceKT128(msg, tk.custom, 100); !bytes.Equal(out, want) {
				errs[i] = fmt.Errorf("size=%d custom=%d: output mismatch", tk.size, len(tk.custom))
			}
		}(i, tk)
	}
	wg.Wait()

	for _, e := range errs {
		if e != nil {
			t.Error(e)
		}
	}
}

// TestConcurrentSharedReadOnly hammers one shared, never-mutated hasher with
// concurrent Clone, Equal, and Pos calls. All three are pure reads, so this must
// be race-free, and the shared hasher must remain unfinalized and unchanged
// afterward. A regression that made Clone or Equal mutate the receiver (e.g. a
// lazy initialization) would both race here and corrupt the shared state.
func TestConcurrentSharedReadOnly(t *testing.T) {
	custom := []byte("ctx")
	msg := ptn(3*BlockSize + 7)
	h := New(custom)
	if _, err := h.Write(msg); err != nil {
		t.Fatalf("Write: %v", err)
	}
	want := referenceKT128(msg, custom, 64)
	wantPos := uint64(len(msg))

	const (
		workers = 16
		rounds  = 50
	)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for range rounds {
				if got := h.Pos(); got != wantPos {
					errs[w] = fmt.Errorf("Pos() = %d, want %d", got, wantPos)
					return
				}
				out := make([]byte, 64)
				_, _ = h.Clone().Read(out) // read the clone, never h itself
				if !bytes.Equal(out, want) {
					errs[w] = fmt.Errorf("clone output mismatch")
					return
				}
				if h.Equal(h) != 1 {
					errs[w] = fmt.Errorf("Equal(h, h) = 0, want 1")
					return
				}
			}
		}(w)
	}
	wg.Wait()

	for _, e := range errs {
		if e != nil {
			t.Error(e)
		}
	}

	// h was only ever cloned/compared, never read, so it is still unfinalized and
	// must produce its original output.
	final := make([]byte, 64)
	_, _ = h.Read(final)
	if !bytes.Equal(final, want) {
		t.Errorf("shared hasher changed after concurrent read-only ops\n got  %x\n want %x", final, want)
	}
}

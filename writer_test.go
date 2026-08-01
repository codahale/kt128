package kt128

import (
	"bufio"
	"bytes"
	"io"
	"testing"
)

func TestClearWriter(t *testing.T) {
	for _, tc := range []struct {
		name  string
		flush bool
	}{
		{name: "unflushed"},
		{name: "flushed", flush: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var dst bytes.Buffer
			w := bufio.NewWriterSize(&dst, 64)
			_, _ = w.Write(bytes.Repeat([]byte{0xA5}, 63))
			if tc.flush {
				_ = w.Flush()
			}

			ClearWriter(w)

			if w.Buffered() != 0 || w.Available() != w.Size() {
				t.Fatalf("writer was not reset: buffered=%d available=%d size=%d", w.Buffered(), w.Available(), w.Size())
			}
			storage := w.AvailableBuffer()
			storage = storage[:cap(storage)]
			if bytes.Count(storage, []byte{0}) != len(storage) {
				t.Fatalf("writer buffer was not zeroed: %x", storage)
			}

			before := dst.Len()
			_, _ = w.Write([]byte("discarded"))
			_ = w.Flush()
			if dst.Len() != before {
				t.Fatal("cleared writer retained its former destination")
			}

			var reused bytes.Buffer
			w.Reset(&reused)
			_, _ = w.WriteString("reused")
			_ = w.Flush()
			if reused.String() != "reused" {
				t.Fatalf("reused writer produced %q", reused.String())
			}
		})
	}
}

func TestClearZeroWriter(t *testing.T) {
	var w bufio.Writer
	ClearWriter(&w)
	if w.Size() == 0 || w.Available() != w.Size() {
		t.Fatalf("zero writer was not initialized: available=%d size=%d", w.Available(), w.Size())
	}
	_, _ = w.WriteString("discarded")
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush after ClearWriter: %v", err)
	}
}

func TestClearWriterDoesNotAllocate(t *testing.T) {
	w := bufio.NewWriterSize(io.Discard, 64)
	allocs := testing.AllocsPerRun(20, func() {
		ClearWriter(w)
	})
	if allocs != 0 {
		t.Fatalf("ClearWriter allocated %.0f times, want 0", allocs)
	}
}

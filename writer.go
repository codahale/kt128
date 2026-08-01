package kt128

import (
	"bufio"
	"io"
)

// ClearWriter discards any unflushed data in w, detaches its destination, and
// makes a best effort to zero its backing buffer. It does not flush w. After
// ClearWriter returns, w writes to [io.Discard]; call [bufio.Writer.Reset] to
// reuse it with another destination.
//
// ClearWriter cannot erase data already written to the former destination,
// copies made by the Go compiler or runtime, or values left in registers.
func ClearWriter(w *bufio.Writer) {
	w.Reset(io.Discard)
	b := w.AvailableBuffer()
	wipeBytes(b[:cap(b)])
}

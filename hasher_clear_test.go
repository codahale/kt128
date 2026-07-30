package kt128

import (
	"bytes"
	"testing"
)

func TestBufferGrowthWipesOldAllocation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		added int
	}{
		{name: "ordinary", added: 9},
		{name: "high-water", added: max(growJumpMin, 9)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := New(nil)
			old := bytes.Repeat([]byte{0xA5}, 16)
			h.buf = old[:8]
			added := bytes.Repeat([]byte{0x5A}, tc.added)

			h.bufferTail(added)

			if bytes.Count(old, []byte{0}) != len(old) {
				t.Fatalf("abandoned buffer was not zeroed: %x", old)
			}
			if !bytes.Equal(h.buf[:8], bytes.Repeat([]byte{0xA5}, 8)) {
				t.Fatalf("grown buffer did not preserve existing data: %x", h.buf[:8])
			}
			if !bytes.Equal(h.buf[8:], added) {
				t.Fatalf("grown buffer tail = %x, want %x", h.buf[8:], added)
			}
			h.Clear()
		})
	}
}

func TestClearPreservesCustomizationAndWipesMessage(t *testing.T) {
	custom := []byte("public domain separator")
	h := New(custom)
	h.buf = bytes.Repeat([]byte{0xA5}, ChunkSize+31)
	messageStorage := h.buf[:cap(h.buf)]
	customStorage := h.c

	h.Clear()

	if bytes.Count(messageStorage, []byte{0}) != len(messageStorage) {
		t.Fatalf("message buffer was not zeroed")
	}
	if !bytes.Equal(h.c, custom) || !bytes.Equal(customStorage, custom) {
		t.Fatalf("Clear changed customization: got %q, want %q", h.c, custom)
	}
	if h.buf != nil || h.final != (sponge{}) || h.pending != (pendingState{}) || h.pendingLen != 0 || h.pos != 0 || h.leafCount != 0 || h.state != stateSingle || h.ds != 0 {
		t.Fatalf("Clear did not reset Hasher state: %#v", h)
	}
}

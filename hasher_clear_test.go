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

func TestClearWipesCustomizationAndMessage(t *testing.T) {
	custom := []byte("secret customization string")
	h := New(custom)
	h.buf = bytes.Repeat([]byte{0xA5}, ChunkSize+31)
	messageStorage := h.buf[:cap(h.buf)]
	customStorage := h.c[:cap(h.c)]

	h.Clear()

	if bytes.Count(messageStorage, []byte{0}) != len(messageStorage) {
		t.Fatalf("message buffer was not zeroed")
	}
	if bytes.Count(customStorage, []byte{0}) != len(customStorage) {
		t.Fatalf("customization string was not zeroed: %x", customStorage)
	}
	if h.buf != nil || h.c != nil || h.final != (sponge{}) || h.pending != (pendingState{}) || h.pendingLen != 0 || h.pos != 0 || h.leafCount != 0 || h.state != stateSingle || h.ds != 0 {
		t.Fatalf("Clear did not reset Hasher state: %#v", h)
	}
}

func TestClearCloneDoesNotWipeSourceCustomization(t *testing.T) {
	custom := []byte("secret customization string")
	h := New(custom)
	clone := h.Clone()
	cloneStorage := clone.c[:cap(clone.c)]

	clone.Clear()

	if bytes.Count(cloneStorage, []byte{0}) != len(cloneStorage) {
		t.Fatalf("clone customization string was not zeroed: %x", cloneStorage)
	}
	if !bytes.Equal(h.c, custom) {
		t.Fatalf("clearing clone changed source customization: got %q, want %q", h.c, custom)
	}

	msg := []byte("message")
	_, _ = h.Write(msg)
	got := make([]byte, 32)
	_, _ = h.Read(got)

	wantHasher := New(custom)
	_, _ = wantHasher.Write(msg)
	want := make([]byte, 32)
	_, _ = wantHasher.Read(want)
	if !bytes.Equal(got, want) {
		t.Fatal("clearing clone changed source output")
	}
}

package kt128

import (
	"bytes"
	"fmt"
	"testing"
)

// This file hardens the Hasher *lifecycle* — the state machine and the
// Clone/Reset/Equal/Write/Read interactions — which FuzzHasher does not reach
// because it only ever drives a single linear Write* -> Read sequence. Here an
// interpreter applies an arbitrary interleaving of operations to a population of
// hashers and checks each one against a model after every step. The model tracks
// only (custom, accumulated message, finalized, bytes squeezed) and derives the
// expected output through referenceKT128, the same independent oracle FuzzHasher
// uses, so the production hasher's lifecycle is validated against an
// implementation that shares no state-management code with it.

const (
	lcMaxMsg     = 3 * BlockSize // per-slot message cap (crosses single->tree and several leaves)
	lcMaxSqueeze = 4096          // per-slot squeeze cap (spans many rate blocks)
	lcMaxSlots   = 8             // bound the live hasher population
)

// lcModel is the reference state of one hasher. The expected output for a hasher
// that has already squeezed `squeezed` bytes is bytes [squeezed, squeezed+n) of
// referenceKT128(msg, custom). Equal compares the next 32 such bytes of each
// operand (it clones and reads 32 bytes), so finalized hashers that have squeezed
// different amounts produce different Equal operands even over the same message.
type lcModel struct {
	custom    []byte
	msg       []byte
	finalized bool
	squeezed  int
}

func (m *lcModel) clone() *lcModel {
	// custom is never mutated after New, so sharing it matches Clone; msg is
	// copied so the two models evolve independently.
	return &lcModel{custom: m.custom, msg: bytes.Clone(m.msg), finalized: m.finalized, squeezed: m.squeezed}
}

// equalBytes returns the 32 output bytes Equal would read from this hasher: the
// slice starting at the current squeeze position.
func (m *lcModel) equalBytes() [32]byte {
	full := referenceKT128(m.msg, m.custom, m.squeezed+32)
	var out [32]byte
	copy(out[:], full[m.squeezed:m.squeezed+32])
	return out
}

func lcModelEqual(a, b *lcModel) int {
	if a.equalBytes() == b.equalBytes() {
		return 1
	}
	return 0
}

type lcSlot struct {
	h *Hasher
	m *lcModel
}

// lcKind enumerates the operations the interpreter applies.
type lcKind int

const (
	lcWrite lcKind = iota
	lcRead
	lcClone
	lcEqual
	lcReset
)

type lcOp struct {
	kind        lcKind
	slot, slot2 int  // operand slots (reduced mod the live slot count)
	length      int  // Write/Read length
	seed        byte // Write payload seed
}

func opWrite(slot, length int, seed byte) lcOp {
	return lcOp{kind: lcWrite, slot: slot, length: length, seed: seed}
}
func opRead(slot, length int) lcOp { return lcOp{kind: lcRead, slot: slot, length: length} }
func opClone(slot, dst int) lcOp   { return lcOp{kind: lcClone, slot: slot, slot2: dst} }
func opEqual(a, b int) lcOp        { return lcOp{kind: lcEqual, slot: a, slot2: b} }
func opReset(slot int) lcOp        { return lcOp{kind: lcReset, slot: slot} }

// lcSeedBytes returns n deterministic pseudorandom bytes (splitmix64) for fuzz
// seeds and the decode test. Defined locally so this file compiles on every
// target, including purego and the non-amd64/arm64 ports where the assembly test
// helpers (and their splitmixBytes) are build-excluded.
func lcSeedBytes(n int, seed uint64) []byte {
	b := make([]byte, n)
	x := seed
	for i := range b {
		x += 0x9E3779B97F4A7C15
		z := x
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		z ^= z >> 31
		b[i] = byte(z)
	}
	return b
}

// lcFill builds a deterministic length-n payload from seed so writes carry
// varying content (catching ordering bugs) while staying reproducible in the
// model, which appends the identical bytes.
func lcFill(seed byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(int(seed)*7 + i*191 + (i >> 8))
	}
	return b
}

// lcWritePanics reports whether Write panicked, recovering it. Used to assert the
// finalized-hasher write contract without aborting the interpreter.
func lcWritePanics(h *Hasher, p []byte) (panicked bool) {
	defer func() {
		if recover() != nil {
			panicked = true
		}
	}()
	_, _ = h.Write(p)
	return false
}

// executeLifecycle runs ops against one hasher per customization string, checking
// every slot against its model after each step.
func executeLifecycle(t *testing.T, customs [][]byte, ops []lcOp) {
	t.Helper()

	var slots []*lcSlot
	for _, c := range customs {
		slots = append(slots, &lcSlot{h: New(c), m: &lcModel{custom: c}})
	}
	if len(slots) == 0 {
		slots = append(slots, &lcSlot{h: New(nil), m: &lcModel{}})
	}

	for step, op := range ops {
		i := op.slot % len(slots)
		s := slots[i]

		switch op.kind {
		case lcWrite:
			l := op.length
			if room := lcMaxMsg - len(s.m.msg); l > room {
				l = room
			}
			payload := lcFill(op.seed, l)
			if s.m.finalized {
				// Writing after the first Read (finalization) must panic and leave
				// state untouched.
				if !lcWritePanics(s.h, payload) {
					t.Fatalf("step %d: Write after finalize on slot %d did not panic", step, i)
				}
			} else {
				if _, err := s.h.Write(payload); err != nil {
					t.Fatalf("step %d: Write: %v", step, err)
				}
				s.m.msg = append(s.m.msg, payload...)
			}

		case lcRead:
			l := op.length
			if room := lcMaxSqueeze - s.m.squeezed; l > room {
				l = room
			}
			got := make([]byte, l)
			if _, err := s.h.Read(got); err != nil {
				t.Fatalf("step %d: Read: %v", step, err)
			}
			var want []byte
			if l > 0 {
				full := referenceKT128(s.m.msg, s.m.custom, s.m.squeezed+l)
				want = full[s.m.squeezed : s.m.squeezed+l]
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("step %d: Read slot=%d squeezed=%d len=%d mismatch\n got  %x\n want %x",
					step, i, s.m.squeezed, l, got, want)
			}
			s.m.finalized = true
			s.m.squeezed += l

		case lcClone:
			ns := &lcSlot{h: s.h.Clone(), m: s.m.clone()}
			// A fresh clone must compare equal to its source.
			if s.h.Equal(ns.h) != 1 {
				t.Fatalf("step %d: clone of slot %d not equal to source", step, i)
			}
			if len(slots) < lcMaxSlots {
				slots = append(slots, ns)
			} else {
				slots[op.slot2%len(slots)] = ns
			}

		case lcEqual:
			j := op.slot2 % len(slots)
			got := slots[i].h.Equal(slots[j].h)
			want := lcModelEqual(slots[i].m, slots[j].m)
			if got != want {
				t.Fatalf("step %d: Equal(slot %d, slot %d) = %d, want %d", step, i, j, got, want)
			}

		case lcReset:
			s.h.Reset()
			s.m.msg = s.m.msg[:0]
			s.m.finalized = false
			s.m.squeezed = 0
		}

		// Pos must always equal the number of message bytes the model has
		// accumulated, for every live slot.
		for k, sl := range slots {
			if sl.h.Pos() != uint64(len(sl.m.msg)) {
				t.Fatalf("step %d: slot %d Pos() = %d, want %d", step, k, sl.h.Pos(), len(sl.m.msg))
			}
		}
	}

	// Final sweep: even slots never observed via Read during the run must still
	// match their model. Cloning avoids disturbing the slot's squeeze position.
	for k, sl := range slots {
		var got [32]byte
		_, _ = sl.h.Clone().Read(got[:])
		if want := sl.m.equalBytes(); got != want {
			t.Fatalf("final: slot %d state diverged from model\n got  %x\n want %x", k, got, want)
		}
	}
}

// lcLenBoundaries clusters Write lengths around the chunk and rate boundaries
// where the state machine changes behavior.
var lcLenBoundaries = []int{
	0, 1, 167, 168, 169,
	BlockSize - 1, BlockSize, BlockSize + 1,
	2 * BlockSize, 2*BlockSize + 1, 3 * BlockSize,
}

func lcDecodeLen(b byte) int {
	if b < 0x60 {
		return int(b) // 0..95: dense small lengths
	}
	return lcLenBoundaries[int(b)%len(lcLenBoundaries)]
}

// decodeOps interprets a fuzz program as an operation stream.
func decodeOps(program []byte) []lcOp {
	const maxOps = 256
	pc := 0
	next := func() byte {
		if pc >= len(program) {
			return 0
		}
		b := program[pc]
		pc++
		return b
	}

	var ops []lcOp
	for pc < len(program) && len(ops) < maxOps {
		// Weight the op distribution toward Write/Read, which drive the state
		// machine, while keeping Clone/Equal/Reset frequent enough to interleave.
		var op lcOp
		switch o := next() % 8; {
		case o <= 2:
			op.kind = lcWrite
		case o <= 4:
			op.kind = lcRead
		case o == 5:
			op.kind = lcClone
		case o == 6:
			op.kind = lcEqual
		default:
			op.kind = lcReset
		}
		op.slot = int(next())
		switch op.kind {
		case lcWrite:
			lb := next()
			op.seed = next()
			op.length = lcDecodeLen(lb)
		case lcRead:
			op.length = int(next())
		case lcClone, lcEqual:
			op.slot2 = int(next())
		}
		ops = append(ops, op)
	}
	return ops
}

// FuzzHasherLifecycle drives an adversarial interleaving of Write, Read, Clone,
// Reset, and Equal across two hashers (with distinct customization strings) and
// their clones, checking every operation against the model. It exercises the
// stateSingle->stateTree->stateFinalized transitions, Clone independence
// (including cloning a mid-squeeze finalized hasher), Reset reuse (including
// reset-after-finalize), the write-after-finalize panic, and Equal's
// squeeze-position semantics — none of which FuzzHasher's linear path reaches.
func FuzzHasherLifecycle(f *testing.F) {
	f.Add([]byte(""), []byte("domain"), []byte{})
	f.Add([]byte("alpha"), []byte("beta"), []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	f.Add([]byte(""), []byte(""), lcSeedBytes(64, 99))
	f.Add([]byte("x"), []byte(""), lcSeedBytes(256, 7))

	f.Fuzz(func(t *testing.T, custom0, custom1, program []byte) {
		if len(custom0) > lcMaxMsg {
			custom0 = custom0[:lcMaxMsg]
		}
		if len(custom1) > lcMaxMsg {
			custom1 = custom1[:lcMaxMsg]
		}
		executeLifecycle(t, [][]byte{custom0, custom1}, decodeOps(program))
	})
}

// TestHasherLifecycle runs hand-built operation sequences through the same
// interpreter so the key transitions are covered deterministically in `go test`,
// independent of the fuzz corpus (fuzzing itself runs only nightly).
func TestHasherLifecycle(t *testing.T) {
	cases := []struct {
		name    string
		customs [][]byte
		ops     []lcOp
	}{
		{
			name:    "tree then write-after-finalize then reset-reuse",
			customs: [][]byte{nil, []byte("d")},
			ops: []lcOp{
				opWrite(0, BlockSize+1, 1), // enter tree mode
				opRead(0, 40),              // finalize + squeeze
				opRead(0, 24),              // continue squeezing
				opWrite(0, 10, 2),          // must panic
				opReset(0),                 // reset a finalized hasher
				opWrite(0, 5, 3),           // reuse
				opRead(0, 32),
			},
		},
		{
			name:    "clone before finalize then diverge",
			customs: [][]byte{[]byte("x"), nil},
			ops: []lcOp{
				opWrite(0, 100, 1),
				opClone(0, 0), // -> slot 2
				opEqual(0, 2), // identical clone: equal
				opWrite(0, 50, 2),
				opEqual(0, 2), // diverged: not equal
				opRead(0, 32),
				opRead(2, 32),
				opEqual(0, 2),
			},
		},
		{
			name:    "clone a mid-squeeze finalized hasher",
			customs: [][]byte{nil, nil},
			ops: []lcOp{
				opWrite(0, 2*BlockSize+5, 7), // tree mode, several leaves
				opRead(0, 13),                // finalize at an odd offset
				opClone(0, 0),                // clone mid-squeeze -> slot 2
				opRead(0, 19),
				opRead(2, 19), // clone resumes the same output stream
				opEqual(0, 2), // both at squeeze position 32: equal
			},
		},
		{
			name:    "equal across different customizations",
			customs: [][]byte{[]byte("alpha"), []byte("beta")},
			ops: []lcOp{
				opWrite(0, 100, 1),
				opWrite(1, 100, 1),
				opEqual(0, 1), // same message, different customization: not equal
				opReset(0),
				opReset(1),
				opEqual(0, 1), // empty message, different customization: not equal
			},
		},
		{
			name:    "single-node interleaved partial reads across rate boundary",
			customs: [][]byte{nil, nil},
			ops: []lcOp{
				opWrite(0, 50, 1),
				opRead(0, 1),
				opRead(0, 7),
				opRead(0, 168), // cross a rate block mid-stream
				opRead(0, 200),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			executeLifecycle(t, tc.customs, tc.ops)
		})
	}
}

// TestDecodeOpsTerminates is a guard that decodeOps always halts and respects its
// op cap, so a fuzz program can never spin the interpreter.
func TestDecodeOpsTerminates(t *testing.T) {
	for _, n := range []int{0, 1, 7, 1000, 4096} {
		ops := decodeOps(lcSeedBytes(n, uint64(n)+1))
		if len(ops) > 256 {
			t.Fatalf("n=%d: decodeOps returned %d ops, want <= 256", n, len(ops))
		}
		_ = fmt.Sprint(ops) // ensure the ops are well-formed values
	}
}

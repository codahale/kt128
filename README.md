# kt128

`kt128` is a Go implementation of KT128 (KangarooTwelve) as specified in RFC 9861.

KT128 is an extendable-output function (XOF) built on TurboSHAKE128. This package supports incremental writes,
arbitrary-length output, customization strings, and optimized tree hashing for large inputs.

## Highlights

- Implements KT128 as a streaming `hash.XOF`.
- Switches to tree mode once the input exceeds one 8192-byte chunk.
- Uses optimized assembly on `amd64` and `arm64`.
- Falls back to pure Go on other targets, or with `-tags purego`.
- Exposes `Clone`, `Reset`, `ClearWriter`, and `Pos` helpers.

## Requirements

- Go `1.26.1` or newer

## Install

```bash
go get github.com/codahale/kt128
```

## Basic Usage

```go
package main

import (
	"encoding/hex"
	"fmt"

	"github.com/codahale/kt128"
)

func main() {
	h := kt128.New(nil)
	_, _ = h.Write([]byte("hello, world"))

	out := make([]byte, 32)
	_, _ = h.Read(out)

	fmt.Println(hex.EncodeToString(out))
}
```

`Read` finalizes the hasher on first use and then continues squeezing output on subsequent calls. Because KT128 is an
XOF, you choose the output length by the size of the destination buffer.

## Security

KT128 targets 128-bit security. Use at least 16 output bytes for 128-bit single-target preimage and second-preimage
resistance, and at least 32 output bytes for 128-bit collision resistance. Multi-target preimage resistance requires
additional output bits as described in RFC 9861 Section 7.

KT128 is designed to be fast and is not a password-hashing function. Use a purpose-built password-hashing function for
passwords and other low-entropy secrets.

## Customization

Construct an immutable customization string and pass it to `New`:

```go
custom := kt128.NewCustomizationString([]byte("example-domain"))
defer custom.Clear()

h := kt128.New(custom)
_, _ = h.Write([]byte("hello, world"))

out := make([]byte, 64)
_, _ = h.Read(out)
```

`NewCustomizationString` copies its input, so the caller may immediately modify or clear the original slice. The
resulting `CustomizationString` is immutable and may be shared by multiple hashers and their clones. `Clear` overwrites
the owned storage and permanently invalidates the customization string. A hasher first finalized after its customization
string has been cleared will panic; a hasher finalized before `Clear` continues squeezing normally. Call `Clear` only
after every absorbing hasher and clone has been finalized or retired; it does not synchronize with finalization.

## Performance Notes

Once the input (the message plus the customization string and its length encoding) exceeds one 8 KiB KT128 chunk, the
implementation switches to tree hashing. Leaf compression is processed in parallel:

- `amd64`: 8-wide AVX-512 kernels for whole batches and masked remainders, with 2-wide AVX-512VL kernels where only
  two lanes are live; AVX2 kernels when AVX-512 is unavailable; generic kernels when neither ISA is available (use
  `GODEBUG=cpu.avx512f=off` to disable AVX-512 and `GODEBUG=cpu.avx2=off` to disable AVX2)
- `arm64` with the SHA3 extension: a hybrid scalar/NEON kernel that compresses five chunks per pass — four on the
  NEON unit and a fifth woven onto the otherwise-idle scalar pipes — with 2-wide NEON kernels draining remainders;
  generic kernels otherwise
- other targets or `purego`: scalar fallback

The first chunk and any trailing partial chunk are fused into the parallel passes rather than absorbed serially, so
throughput holds across ragged message sizes. Representative one-shot throughput at 1 MiB: ~6.6 GB/s on an Apple M4
Pro and ~6.7 GB/s on Intel Emerald Rapids (~2.2 GB/s on the AVX2 kernels with AVX-512 disabled).

## API Notes

- `NewCustomizationString(c)` defensively copies `c` into an immutable, shareable customization string.
- `CustomizationString.Clear` wipes its owned storage and permanently invalidates the customization string. It must not
  run concurrently with finalization or another call to `Clear`.
- `New(c)` creates a new hasher sharing `c` (pass nil for none).
- `Write` absorbs message bytes without retaining the input slice.
- `Read(dst)` squeezes output into `dst`.
- `Clone` copies the current hashing state but shares its immutable customization string.
- `Reset` makes a best effort to zero message-dependent state and resets the hasher for reuse while preserving its
  customization string; resetting after that string is cleared makes the next `Read` panic.
- `ClearWriter` discards pending data from a `bufio.Writer`, detaches its destination, and makes a best effort to zero
  its backing buffer.
- `RecommendedWriteBufferSize` reports a runtime dispatch-specific buffer size for coalescing small writes into
  parallel leaf batches.
- `Pos` returns the number of bytes written so far. `Write` panics before the message length would reach 2^64 bytes
  without an intervening `Reset`.
- `Hasher.BlockSize()` reports the 168-byte TurboSHAKE128 sponge rate; `ChunkSize` is the 8192-byte KT128 tree chunk.

## Ownership and Buffering

The caller owns every input and output slice. `Write` absorbs its argument before returning and does not retain it, so
the caller may immediately modify or reuse a message slice. `Read` writes directly into its argument and does not retain
it. A `Hasher` retains only fixed-size hashing state and a pointer to its `CustomizationString`; it does not retain
message bytes or allocate a message-sized internal buffer. The `CustomizationString` owns a defensive copy of its input.

Complete leaves contiguous within a `Write` use the parallel kernels, while leaves assembled from smaller writes are
absorbed incrementally. Applications issuing small writes can recover bulk throughput with an explicitly sized buffer;
the default 4 KiB `bufio.Writer` is too small for this purpose:

```go
h := kt128.New(custom)
w := bufio.NewWriterSize(h, kt128.RecommendedWriteBufferSize())
```

The recommendation is eight chunks on amd64 with AVX2 or AVX-512, five chunks on arm64 with the SHA3 extension, and
one chunk on scalar implementations. Larger multiples may be used when retaining more message data is acceptable.

External buffering remains entirely caller managed. A `bufio.Writer` may retain message bytes in its backing array;
large writes may instead pass directly through to the Hasher. Bytes reported by `w.Buffered()` have not reached the
Hasher. Call `w.Flush()` before `Read`, `Clone`, or `Pos` when those operations must account for every submitted byte.
Before `Reset`, either flush pending bytes if they belong to the current message or discard them with `ClearWriter`.
The caller owns the writer, its destination, and the sequencing and error handling for `Write` and `Flush`.

`ClearWriter` deliberately discards rather than flushes pending bytes, detaches the destination by resetting the writer
to `io.Discard`, and makes a best effort to wipe the writer's backing array. It does not clear bytes already flushed to
the former destination. After producing the final output, clear caller-owned storage only after every object retaining
it has been retired:

```go
kt128.ClearWriter(w)
h.Reset()
// Retire h and every clone here, then clear the shared customization.
custom.Clear()
```

As with any best-effort clearing operation in Go, `Reset`, `ClearWriter`, and `CustomizationString.Clear` cannot erase
copies made by the compiler or runtime, data already written elsewhere, or values left in registers.

## License

Dual-licensed under Apache-2.0 and MIT. See LICENSE-APACHE and LICENSE-MIT.

# kt128

`kt128` is a Go implementation of KT128 (KangarooTwelve) as specified in RFC 9861.

KT128 is an extendable-output function (XOF) built on TurboSHAKE128. This package supports incremental writes,
arbitrary-length output, customization strings, and optimized tree hashing for large inputs.

## Highlights

- Implements KT128 as a streaming `hash.XOF` and as a `hash.Hash` with a 256-bit digest.
- Switches to tree mode once the input exceeds one 8192-byte chunk.
- Uses optimized assembly on `amd64` and `arm64`.
- Falls back to pure Go on other targets, or with `-tags purego`.
- Exposes `Clone`, `Reset`, and `Pos` helpers.

## Requirements

- Go `1.26.1` or newer

## Install

```bash
go get github.com/codahale/kt128
```

## Basic Usage

For the common 32-byte output:

```go
sum := kt128.Sum256([]byte("hello, world"), nil)
fmt.Printf("%x\n", sum)
```

For incremental input or arbitrary-length output:

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

This package does not perform security-motivated zeroing of hashing state and provides no secure-erasure guarantee.
`Reset` reinitializes a Hasher for reuse but does not guarantee that previous state is unrecoverable. This matches the
Go standard library's
[`crypto/sha256`](https://pkg.go.dev/crypto/sha256) and [`crypto/hkdf`](https://pkg.go.dev/crypto/hkdf) APIs, which expose
no state-zeroing operation.

## Customization

Pass the customization string to `New`:

```go
custom := []byte("example-domain")
h := kt128.New(custom)

_, _ = h.Write([]byte("hello, world"))

out := make([]byte, 64)
_, _ = h.Read(out)
```

`New` copies its customization input, so the caller may immediately modify or reuse the original slice. `Clone` copies
the customization again, making each Hasher independently owned.

## Performance Notes

Once the input (the message plus the customization string and its length encoding) exceeds one 8 KiB KT128 chunk, the
implementation switches to tree hashing. Leaf compression is processed in parallel:

- `amd64`: 8-wide AVX-512 kernels for whole batches and masked remainders, with 2-wide AVX-512VL kernels where only
  two lanes are live; AVX2 kernels when AVX-512 is unavailable; generic kernels when neither ISA is available
- `arm64` with the SHA3 extension: a hybrid scalar/NEON kernel that compresses five chunks per pass — four on the
  NEON unit and a fifth woven onto the otherwise-idle scalar pipes — with 2-wide NEON kernels draining remainders;
  generic kernels otherwise
- other targets or `purego`: scalar fallback

The first chunk and any trailing partial chunk are fused into the parallel passes rather than absorbed serially, so
throughput holds across ragged message sizes. Representative one-shot throughput at 1 MiB: ~6.6 GB/s on an Apple M4
Pro and ~6.7 GB/s on Intel Emerald Rapids (~2.2 GB/s on the AVX2 kernels with AVX-512 disabled).

## Assembly Dispatch

The package detects CPU features once during process initialization and selects the fastest supported implementation.
There is no exported dispatch API and no supported way to force instructions that the host CPU or operating system does
not report as available.

On `amd64`, leaf hashing and single-sponge operations use separate dispatch ladders:

- Leaf hashing uses AVX-512 when AVX-512, AVX-512F, and AVX-512VL are available; otherwise AVX2; otherwise generic Go.
- Sponge permutation and absorption use AVX-512 when available; otherwise BMI2 assembly; otherwise generic Go.

On `arm64`, the SHA3 extension gates both the NEON leaf kernels and the assembly sponge implementation. Without SHA3,
both paths use generic Go. Other architectures always use generic Go.

Go's `GODEBUG=cpu.<feature>=off` settings disable individual runtime-detected features. They affect the entire process,
not only this package, and cannot enable unsupported features. Common configurations are:

| Configuration | Setting | Result |
| --- | --- | --- |
| Disable AVX-512 on `amd64` | `GODEBUG=cpu.avx512f=off` | AVX2 leaves and BMI2 sponge assembly remain available independently. |
| Force generic Go on `amd64` | `GODEBUG=cpu.avx512f=off,cpu.avx2=off,cpu.bmi2=off` | Disables every assembly dispatch path used by this package. |
| Force generic Go on `arm64` | `GODEBUG=cpu.sha3=off` | Disables the SHA3-gated leaf and sponge assembly paths. |

For example:

```bash
GODEBUG=cpu.avx512f=off go test ./...
GODEBUG=cpu.avx512f=off,cpu.avx2=off,cpu.bmi2=off go test ./...
GODEBUG=cpu.sha3=off go test ./...
```

For a build that excludes architecture-specific assembly entirely, use the `purego` build tag:

```bash
go test -tags purego ./...
go build -tags purego .
```

`RecommendedWriteBufferSize` reflects the active leaf dispatch: eight chunks for AVX-512 or AVX2, five chunks for
arm64 SHA3, and one chunk for generic Go.

## API Notes

- `Sum256(message, customization)` returns a 32-byte hash without retaining either input slice.
- `New(c)` creates a new hasher with a defensive copy of `c` (pass nil for none).
- `Write` absorbs message bytes without retaining the input slice.
- `Sum` appends a 32-byte digest without changing the absorption state. It panics after `Read` finalizes the hasher.
- `Read(dst)` squeezes output into `dst`.
- `Clone` implements `hash.Cloner`, returning an independent copy of the current hashing state and customization string.
- `Reset` reinitializes the hasher for reuse while preserving its customization string; it does not guarantee erasure
  of the previous hashing state.
- `RecommendedWriteBufferSize` reports a runtime dispatch-specific buffer size for coalescing small writes into
  parallel leaf batches.
- `Pos` returns the number of bytes written so far. `Write` panics before the message length would reach 2^64 bytes
  without an intervening `Reset`.
- `Hasher.BlockSize()` reports the 168-byte TurboSHAKE128 sponge rate; `ChunkSize` is the 8192-byte KT128 tree chunk.

## Ownership and Buffering

The caller owns every input and output slice. `Write` absorbs its argument before returning and does not retain it, so
the caller may immediately modify or reuse a message slice. `Read` writes directly into its argument and does not retain
it. A `Hasher` retains fixed-size hashing state and an owned copy of its customization string; it does not retain
message bytes or allocate a message-sized internal buffer.

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
Before `Reset`, either flush pending bytes if they belong to the current message or reset the writer without flushing
if they should be discarded.
The caller owns the writer, its destination, and the sequencing and error handling for `Write` and `Flush`.

Neither resetting a `Hasher` nor resetting a `bufio.Writer` promises to erase its previous contents. The Go compiler
and runtime may also retain copies in stack slots, registers, or other runtime-managed storage.

## License

Dual-licensed under Apache-2.0 and MIT. See LICENSE-APACHE and LICENSE-MIT.

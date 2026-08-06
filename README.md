# kt128

`kt128` is a Go implementation of KT128 (KangarooTwelve) as specified in
[RFC 9861](https://www.rfc-editor.org/rfc/rfc9861.html).

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

## Development

The repository's check scripts require [`actionlint`](https://github.com/rhysd/actionlint) and
[`golangci-lint`](https://golangci-lint.run/) v2 in addition to Go. Run the checks directly with:

```bash
script/check-commit
script/check-push
```

The repository includes matching pre-commit and pre-push hooks. Enable them for the clone with:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook runs `script/check-commit` against the staged tree. The pre-push hook runs
`script/check-push` against each commit being pushed.

## Install

```bash
go get github.com/codahale/kt128
```

## Basic Usage

For one-shot hashing with a 32-byte output:

```go
sum := kt128.Sum([]byte("hello, world"), nil, 32)
fmt.Printf("%x\n", sum)
```

For incremental input with a fixed 32-byte digest:

```go
h := kt128.New(nil)
_, _ = h.Write([]byte("hello, "))
_, _ = h.Write([]byte("world"))
sum := h.Sum(nil)
```

`Sum` appends the digest to its argument without finalizing the hasher, so more input may be written afterward.

For arbitrary-length XOF output:

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

Any call to `Read`, including a zero-length call, finalizes the hasher. Subsequent reads continue squeezing the same
output stream; `Write` and `Sum` panic after finalization. Because KT128 is an XOF, an arbitrary
number of `Read` calls can be made for an arbitrary number of output bytes.

## Security

KT128 targets 128-bit security. Use at least 16 output bytes for 128-bit single-target preimage and second-preimage
resistance, and at least 32 output bytes for 128-bit collision resistance. Multi-target preimage resistance may require
additional output bits as described in RFC 9861 Section 7.

KT128 is designed to be fast and is not a password-hashing function. Use a purpose-built password-hashing function for
passwords and other low-entropy secrets.

This package does not perform security-motivated zeroing of hashing state and provides no secure-erasure guarantee.
`Reset` reinitializes a Hasher for reuse but does not guarantee that previous state is unrecoverable. This matches the
Go standard library's
[`crypto/sha256`](https://pkg.go.dev/crypto/sha256) and [`crypto/hkdf`](https://pkg.go.dev/crypto/hkdf) APIs, which expose
no state-zeroing operation.

The binary state returned by `MarshalBinary` or `AppendBinary` contains the customization string and resumable internal
hash state. The encoding provides neither confidentiality nor authenticity; protect and authenticate it externally
when either property is required. `UnmarshalBinary` validates the format and all derivable state invariants, but cannot
establish that otherwise valid sponge lanes originated from a particular message.

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

- `amd64`: 8-wide AVX-512 kernels for whole batches and 5–7-leaf remainders, 4-wide kernels for 3–4 leaves, and 2-wide
  kernels for two leaves; AVX2 kernels when AVX-512 is unavailable; generic kernels when neither ISA is available
- `arm64` with the SHA3 extension: a hybrid scalar/NEON kernel that compresses five chunks per pass — four on the
  NEON unit and a fifth woven onto the otherwise-idle scalar pipes — with 2-wide NEON kernels draining remainders;
  generic kernels otherwise
- other targets or `purego`: scalar fallback

On accelerated paths, the first chunk and a trailing partial chunk are fused into parallel passes when a suitable
kernel is available, preserving throughput across ragged message sizes. Representative one-shot throughput at 1 MiB:
~6.6 GB/s on an Apple M4 Pro and ~6.7 GB/s on Intel Emerald Rapids (~2.2 GB/s on the AVX2 kernels with AVX-512
disabled).

## Assembly Dispatch

The package detects CPU features once during process initialization and selects the fastest supported implementation.
There is no exported dispatch API and no supported way to force instructions that the host CPU or operating system does
not report as available.

On `amd64`, leaf hashing and single-sponge operations use separate dispatch ladders:

- Leaf hashing uses AVX-512 when AVX-512, AVX-512F, AVX-512VL, and AVX-512DQ are available; otherwise AVX2; otherwise
  generic Go.
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

- `Sum(message, customization, outputLen)` returns the requested number of hash bytes without retaining either input
  slice. It panics if `outputLen` is negative.
- `New(c)` creates a new hasher with a defensive copy of `c` (pass nil for none).
- `Write` absorbs message bytes without retaining the input slice.
- `(*Hasher).Sum` appends a 32-byte digest without changing the absorption state. It panics after `Read` finalizes the
  hasher.
- `Read(dst)` finalizes the hasher and squeezes output into `dst`; subsequent reads continue the output stream.
- `Clone() (hash.Cloner, error)` returns an independent copy at the current absorption or squeeze position, including
  an independent copy of the customization string. The dynamic result is a `*Hasher`, and the error is always `nil`.
- `MarshalBinary`, `AppendBinary`, and `UnmarshalBinary` persist and restore the complete state at any absorption or
  squeeze position, including the customization string. Failed unmarshaling leaves the receiver unchanged.
- `Reset` reinitializes the hasher for reuse while preserving its customization string; it does not guarantee erasure
  of the previous hashing state.
- `RecommendedWriteBufferSize` reports a runtime dispatch-specific buffer size for coalescing small writes into
  parallel leaf batches.
- `Pos` returns the number of bytes written so far. `Write` panics before the message length would reach 2^64 bytes
  without an intervening `Reset`.
- `Size` and `Hasher.Size()` report the 32-byte fixed digest size. `Hasher.BlockSize()` reports the 168-byte
  TurboSHAKE128 sponge rate; `ChunkSize` is the 8192-byte KT128 tree chunk.

### Binary State Format

The binary state format is stable across package releases. Version 1 has this layout:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 5 | ASCII identifier `kt128` |
| 5 | 1 | Format version, `1` |
| 6 | 1 | Lifecycle: `0` single-node absorption, `1` tree absorption, `2` finalized |
| 7 | 8 | Message position, unsigned big endian |
| 15 | 8 | Customization length, unsigned big endian |
| 23 | 200 | Final-node Keccak lanes, 25 unsigned little-endian 64-bit words |
| 223 | 1 | Final-node sponge position |
| 224 | 200 | Leaf Keccak lanes, 25 unsigned little-endian 64-bit words |
| 424 | 1 | Leaf sponge position |
| 425 | variable | Customization bytes |

The encoding is canonical: its total size must equal 425 plus the encoded customization length, and lifecycle-specific
positions and inactive leaf state must agree with the message position.

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
Hasher. Call `w.Flush()` before `Sum`, `Read`, `Clone`, or `Pos` when those operations must account for every submitted
byte. Before `Reset`, either flush pending bytes if they belong to the current message or reset the writer without
flushing if they should be discarded.
The caller owns the writer, its destination, and the sequencing and error handling for `Write` and `Flush`.

Neither resetting a `Hasher` nor resetting a `bufio.Writer` promises to erase its previous contents. The Go compiler
and runtime may also retain copies in stack slots, registers, or other runtime-managed storage.

## License

Dual-licensed under Apache-2.0 and MIT. See LICENSE-APACHE and LICENSE-MIT.

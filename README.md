# kt128

`kt128` is a Go implementation of KT128 (KangarooTwelve) as specified in RFC 9861.

KT128 is an extendable-output function (XOF) built on TurboSHAKE128. This package supports incremental writes,
arbitrary-length output, customization strings, and optimized tree hashing for large inputs.

## Highlights

- Implements KT128 as a streaming `hash.XOF`.
- Switches to tree mode once the input exceeds one 8192-byte chunk.
- Uses optimized assembly on `amd64` and `arm64`.
- Falls back to pure Go on other targets, or with `-tags purego`.
- Exposes `Clone`, `Reset`, `Equal`, and `Pos` helpers.

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

## Customization

Pass a customization string to `New`:

```go
h := kt128.New([]byte("example-domain"))
_, _ = h.Write([]byte("hello, world"))

out := make([]byte, 64)
_, _ = h.Read(out)
```

## Performance Notes

Once the input (the message plus the customization string and its length encoding) exceeds one 8 KiB KT128 chunk, the
implementation switches to tree hashing. Leaf compression is processed in parallel:

- `amd64`: 8-wide AVX-512 kernels for whole batches and masked remainders, with 2-wide AVX-512VL kernels where only
  two lanes are live; AVX2 kernels otherwise (use the `kt128_disable_avx512` build tag to disable AVX-512)
- `arm64`: a hybrid scalar/NEON kernel that compresses five chunks per pass — four on the NEON unit and a fifth
  woven onto the otherwise-idle scalar pipes — with 2-wide NEON kernels draining remainders
- other targets or `purego`: scalar fallback

The first chunk and any trailing partial chunk are fused into the parallel passes rather than absorbed serially, so
throughput holds across ragged message sizes. Representative one-shot throughput at 1 MiB: ~6.6 GB/s on an Apple M4
Pro and ~6.7 GB/s on Intel Emerald Rapids (~2.2 GB/s on the AVX2 kernels with AVX-512 disabled).

## API Notes

- `New(c)` creates a new hasher with customization string `c` (pass nil for none); it copies `c`.
- `Write` absorbs message bytes.
- `Read(dst)` squeezes output into `dst`.
- `Clone` copies the current state so both hashers can evolve independently.
- `Reset` resets the hasher for reuse without scrubbing buffered message data.
- `Clear` zeros message-derived state and resets the hasher for reuse while preserving its customization string.
- `Equal` reports whether two hashers have identical state, returning 1 if equal and 0 otherwise, in constant time.
- `Pos` returns the number of bytes written so far.

## License

Dual-licensed under Apache-2.0 and MIT. See LICENSE-APACHE and LICENSE-MIT.

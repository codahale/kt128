# kt128

`kt128` is a Go implementation of KT128 (KangarooTwelve) as specified in RFC 9861.

KT128 is an extendable-output function (XOF) built on TurboSHAKE128. This package supports incremental writes,
arbitrary-length output, customization strings, and optimized tree hashing for large inputs.

## Highlights

- Implements KT128 as a streaming `hash.XOF`.
- Switches to tree mode for messages larger than 8192 bytes.
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

For messages larger than one 8 KiB KT128 chunk, the implementation switches to tree hashing. Leaf compression is
processed in parallel:

- `amd64`: AVX2, with AVX-512 when available (use the `kt128_disable_avx512` build tag to disable AVX-512)
- `arm64`: optimized assembly path
- other targets or `purego`: scalar fallback

## API Notes

- `New(c)` creates a new hasher with customization string `c` (pass nil for none); it copies `c`.
- `Write` absorbs message bytes.
- `Read(dst)` squeezes output into `dst`.
- `Clone` copies the current state so both hashers can evolve independently.
- `Reset` clears the hasher for reuse.
- `Equal` compares two hasher states in constant time.
- `Pos` returns the number of bytes written so far.

## License

Dual-licensed under Apache-2.0 and MIT. See LICENSE-APACHE and LICENSE-MIT.

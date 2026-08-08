package kt128_test

import (
	"fmt"
	"hash"

	"github.com/codahale/kt128"
)

var (
	_ hash.Hash   = (*kt128.Hash)(nil)
	_ hash.Cloner = (*kt128.Hash)(nil)
	_ hash.XOF    = (*kt128.XOF)(nil)
)

func ExampleNewHash() {
	h := kt128.NewHash(nil)
	_, _ = h.Write([]byte{0})
	fmt.Printf("%x\n", h.Sum(nil))

	// Output:
	// 2bda92450e8b147f8a7cb629e784a058efca7cf7d8218e02d345dfaa65244a1f
}

func ExampleNewXOF() {
	x := kt128.NewXOF(nil)
	_, _ = x.Write([]byte{0})

	var first, second [16]byte
	_, _ = x.Read(first[:])
	_, _ = x.Read(second[:])

	fmt.Printf("%x\n", first)
	fmt.Printf("%x\n", second)

	// Output:
	// 2bda92450e8b147f8a7cb629e784a058
	// efca7cf7d8218e02d345dfaa65244a1f
}

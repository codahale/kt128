package kt128

// Sum returns outputLen bytes of the KT128 hash of message using customization
// as the customization string. Sum does not retain or modify either input
// slice. Sum panics if outputLen is negative.
func Sum(message, customization []byte, outputLen int) []byte {
	if outputLen < 0 {
		panic("kt128: negative output length")
	}

	sum := make([]byte, outputLen)
	var h XOF
	h.c = customization
	_, _ = h.Write(message)
	_, _ = h.Read(sum)
	return sum
}

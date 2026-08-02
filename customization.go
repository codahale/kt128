package kt128

// CustomizationString is an immutable KT128 customization string. Construct
// one with [NewCustomizationString], which copies its input.
//
// A CustomizationString may be shared by any number of Hashers. Clear
// overwrites the owned storage and permanently invalidates the
// CustomizationString. It must only be called after every referring Hasher has
// been finalized or retired. A Hasher first finalized after its customization
// string has been cleared will panic.
//
// A CustomizationString must not be copied after first use.
type CustomizationString struct {
	noCopy  noCopy
	data    []byte
	cleared bool
}

// NewCustomizationString returns an immutable customization string containing
// a copy of p. The caller may modify or clear p immediately after this function
// returns.
func NewCustomizationString(p []byte) *CustomizationString {
	return &CustomizationString{data: append([]byte(nil), p...)}
}

// Clear makes a best effort to overwrite the customization string's owned
// storage and permanently invalidates it. It must not be called concurrently
// with Hasher finalization or another call to Clear. Clear may be called more
// than once sequentially.
//
// As with any best-effort clearing operation in Go, Clear cannot erase copies
// made by the compiler or runtime or values left in registers.
func (c *CustomizationString) Clear() {
	if c == nil {
		return
	}
	if c.cleared {
		return
	}
	wipeBytes(c.data)
	c.data = nil
	c.cleared = true
}

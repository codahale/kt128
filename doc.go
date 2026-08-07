// Package kt128 implements KT128 (KangarooTwelve) as specified in [RFC 9861].
//
// KT128 is a tree-hash eXtendable-Output Function (XOF) built on TurboSHAKE128.
// [Hash] implements [hash.Hash] with a caller-selected digest size, while [XOF]
// implements [hash.XOF] for arbitrary-length output. Create one with [NewHash]
// or [NewXOF], then absorb the message with Write. Reading from an XOF finalizes
// its message; subsequent reads continue the same output stream.
//
// When the input (the message plus the customization string and its length
// encoding) exceeds [ChunkSize] bytes, it splits the input into chunks and
// computes a leaf chain value from each. On amd64 and arm64 the leaves are
// computed in parallel using SIMD-accelerated Keccak permutations when the
// required CPU features are available; other targets and the purego build use
// a scalar fallback.
//
// # Security
//
// KT128 targets 128-bit security. Outputs should be at least 16 bytes for
// 128-bit single-target preimage and second-preimage resistance, and at least
// 32 bytes for 128-bit collision resistance. Multi-target preimage resistance
// may require additional output bits; see RFC 9861 Section 7.
//
// KT128 is designed to be fast and is not a password-hashing function. Use a
// purpose-built password-hashing function for passwords and other low-entropy
// secrets.
//
// This package does not perform security-motivated zeroing of hashing state and
// provides no secure-erasure guarantee. Reset reinitializes a Hash or XOF for
// reuse but does not guarantee that previous state is unrecoverable. This
// matches the Go standard library's [crypto/sha256] and [crypto/hkdf] APIs,
// which expose no state-zeroing operation.
//
// Binary encodings of Hash and XOF values contain their customization strings
// and resumable internal states. They provide neither confidentiality nor
// authenticity.
//
// [RFC 9861]: https://www.rfc-editor.org/rfc/rfc9861.html
package kt128

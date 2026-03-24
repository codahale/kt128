//go:build !amd64 || purego || kt128_disable_avx512

package cpuid

var HasAVX512 = false

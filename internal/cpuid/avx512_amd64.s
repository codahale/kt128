//go:build amd64 && !purego

#include "textflag.h"

// func hasAVX2() bool
TEXT ·hasAVX2(SB), NOSPLIT, $0-1
	// CPUID leaf 7 is required for the AVX2 feature bit.
	XORL	AX, AX
	CPUID
	CMPL	AX, $7
	JL	avx2_no

	// Check AVX and OSXSAVE before using XGETBV.
	MOVL	$1, AX
	XORL	CX, CX
	CPUID
	BTL	$28, CX
	JCC	avx2_no
	BTL	$27, CX
	JCC	avx2_no

	// The OS must save both XMM and YMM state.
	XORL	CX, CX
	XGETBV
	ANDL	$0x06, AX
	CMPL	AX, $0x06
	JNE	avx2_no

	// Check AVX2 (CPUID.7.0:EBX bit 5).
	MOVL	$7, AX
	XORL	CX, CX
	CPUID
	BTL	$5, BX
	JCC	avx2_no

	MOVB	$1, ret+0(FP)
	RET

avx2_no:
	MOVB	$0, ret+0(FP)
	RET

// func hasAVX512VL() bool
TEXT ·hasAVX512VL(SB), NOSPLIT, $0-1
	// CPUID leaf 7 is required for the AVX-512 feature bits.
	XORL	AX, AX
	CPUID
	CMPL	AX, $7
	JL	no

	// Check AVX and OSXSAVE before using XGETBV.
	MOVL	$1, AX
	XORL	CX, CX
	CPUID
	BTL	$28, CX
	JCC	no
	BTL	$27, CX
	JCC	no

	// Check XCR0: OS saves XMM (bit 1), YMM (bit 2), and ZMM
	// state (bits 5, 6, and 7).
	XORL	CX, CX
	XGETBV
	ANDL	$0xE6, AX
	CMPL	AX, $0xE6
	JNE	no

	// Check AVX512F (leaf 7, EBX bit 16) and AVX512VL (EBX bit 31).
	MOVL	$7, AX
	XORL	CX, CX
	CPUID
	BTL	$16, BX
	JCC	no
	BTL	$31, BX
	JCC	no

	MOVB	$1, ret+0(FP)
	RET

no:
	MOVB	$0, ret+0(FP)
	RET

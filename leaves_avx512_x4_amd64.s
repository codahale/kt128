// 4-wide KT128 leaf quad kernels — AVX-512VL (YMM) implementation.
//
// Transcriptions of the 8-wide masked kernels at YMM width: 256-bit EVEX ops
// issue on three vector ALU ports where 512-bit ops get two, so a quad pass
// costs ~0.70 of a masked 8-wide pass (measured Emerald Rapids) while hosting
// up to four lanes. Scheduling routes 3..4-lane shapes here — sub-batch
// remainders, S_0 fusion for 3..4-chunk first writes, and small tail fusions —
// where the 8-wide pass would run mostly empty; 2-lane shapes keep the XMM
// pair (equal pass cost, no masking setup) and 5-plus-lane shapes the 8-wide
// kernels. Structure mirrors leaves_avx_amd64.s kernel-for-kernel: clamped
// gather indices steer dummy lanes to chunk 0 so they read in-bounds memory.

//go:build !purego

#include "textflag.h"
#include "keccak_round_avx512_x4_amd64.h"

// ABSORB_LANE_X4_GATHER gathers one uint64 from 4 instances at the given byte
// offset from BX (data base pointer) using Y28 as the index vector, and XORs
// the result into Ylane. K1 is reset before each gather (gathers consume it).
#define ABSORB_LANE_X4_GATHER(offset, Ylane) \
	KXNORB	K1, K1, K1; \
	VPGATHERQQ	offset(BX)(Y28*1), K1, Y25; \
	VPXORQ	Y25, Ylane, Ylane

// VPBROADCASTQ_IMM_0x0B_X4 broadcasts 0x0B to all 4 qwords in Ydst.
#define VPBROADCASTQ_IMM_0x0B_X4(Ydst) \
	MOVQ	$0x0B, AX; \
	VPBROADCASTQ	AX, Ydst

// VPBROADCASTQ_IMM_0x80_HIGH_X4 broadcasts 0x8000000000000000 to all 4 qwords in Ydst.
#define VPBROADCASTQ_IMM_0x80_HIGH_X4(Ydst) \
	MOVQ	$0x8000000000000000, AX; \
	VPBROADCASTQ	AX, Ydst

// ZERO_STATE_X4 zeroes the Keccak state Y0-Y24.
#define ZERO_STATE_X4 \
	VPXORQ	Y0, Y0, Y0; \
	VPXORQ	Y1, Y1, Y1; \
	VPXORQ	Y2, Y2, Y2; \
	VPXORQ	Y3, Y3, Y3; \
	VPXORQ	Y4, Y4, Y4; \
	VPXORQ	Y5, Y5, Y5; \
	VPXORQ	Y6, Y6, Y6; \
	VPXORQ	Y7, Y7, Y7; \
	VPXORQ	Y8, Y8, Y8; \
	VPXORQ	Y9, Y9, Y9; \
	VPXORQ	Y10, Y10, Y10; \
	VPXORQ	Y11, Y11, Y11; \
	VPXORQ	Y12, Y12, Y12; \
	VPXORQ	Y13, Y13, Y13; \
	VPXORQ	Y14, Y14, Y14; \
	VPXORQ	Y15, Y15, Y15; \
	VPXORQ	Y16, Y16, Y16; \
	VPXORQ	Y17, Y17, Y17; \
	VPXORQ	Y18, Y18, Y18; \
	VPXORQ	Y19, Y19, Y19; \
	VPXORQ	Y20, Y20, Y20; \
	VPXORQ	Y21, Y21, Y21; \
	VPXORQ	Y22, Y22, Y22; \
	VPXORQ	Y23, Y23, Y23; \
	VPXORQ	Y24, Y24, Y24

// GATHER_STRIPE21_X4 reloads the gather index vector from SP+0 and absorbs one
// full 168-byte stripe (21 lanes).
#define GATHER_STRIPE21_X4 \
	VMOVDQU64	0(SP), Y28; \
	ABSORB_LANE_X4_GATHER(0*8, Y0); \
	ABSORB_LANE_X4_GATHER(1*8, Y1); \
	ABSORB_LANE_X4_GATHER(2*8, Y2); \
	ABSORB_LANE_X4_GATHER(3*8, Y3); \
	ABSORB_LANE_X4_GATHER(4*8, Y4); \
	ABSORB_LANE_X4_GATHER(5*8, Y5); \
	ABSORB_LANE_X4_GATHER(6*8, Y6); \
	ABSORB_LANE_X4_GATHER(7*8, Y7); \
	ABSORB_LANE_X4_GATHER(8*8, Y8); \
	ABSORB_LANE_X4_GATHER(9*8, Y9); \
	ABSORB_LANE_X4_GATHER(10*8, Y10); \
	ABSORB_LANE_X4_GATHER(11*8, Y11); \
	ABSORB_LANE_X4_GATHER(12*8, Y12); \
	ABSORB_LANE_X4_GATHER(13*8, Y13); \
	ABSORB_LANE_X4_GATHER(14*8, Y14); \
	ABSORB_LANE_X4_GATHER(15*8, Y15); \
	ABSORB_LANE_X4_GATHER(16*8, Y16); \
	ABSORB_LANE_X4_GATHER(17*8, Y17); \
	ABSORB_LANE_X4_GATHER(18*8, Y18); \
	ABSORB_LANE_X4_GATHER(19*8, Y19); \
	ABSORB_LANE_X4_GATHER(20*8, Y20)

// GATHER_FINAL16_X4 reloads the gather index vector and absorbs the final
// 128-byte partial block (16 lanes).
#define GATHER_FINAL16_X4 \
	VMOVDQU64	0(SP), Y28; \
	ABSORB_LANE_X4_GATHER(0*8, Y0); \
	ABSORB_LANE_X4_GATHER(1*8, Y1); \
	ABSORB_LANE_X4_GATHER(2*8, Y2); \
	ABSORB_LANE_X4_GATHER(3*8, Y3); \
	ABSORB_LANE_X4_GATHER(4*8, Y4); \
	ABSORB_LANE_X4_GATHER(5*8, Y5); \
	ABSORB_LANE_X4_GATHER(6*8, Y6); \
	ABSORB_LANE_X4_GATHER(7*8, Y7); \
	ABSORB_LANE_X4_GATHER(8*8, Y8); \
	ABSORB_LANE_X4_GATHER(9*8, Y9); \
	ABSORB_LANE_X4_GATHER(10*8, Y10); \
	ABSORB_LANE_X4_GATHER(11*8, Y11); \
	ABSORB_LANE_X4_GATHER(12*8, Y12); \
	ABSORB_LANE_X4_GATHER(13*8, Y13); \
	ABSORB_LANE_X4_GATHER(14*8, Y14); \
	ABSORB_LANE_X4_GATHER(15*8, Y15)

// PERMUTE12_X4 runs the 12-round Keccak-p[1600,12] permutation on Y0-Y24.
#define PERMUTE12_X4 \
	LEAQ	kt128_round_consts_2x+192(SB), R11; \
	X4_4ROUNDS(0, 16, 32, 48); \
	X4_4ROUNDS(64, 80, 96, 112); \
	X4_4ROUNDS(128, 144, 160, 176)

// func processLeavesQuadAVX512(input *byte, cvs *byte, n uint64)
//
// Direct-read quad kernel: processes n (2..4) contiguous 8192-byte chunks in a
// single 4-wide pass, writing n×32-byte CVs to cvs. The gather index vector
// maps lanes 0..n-1 to their chunks and clamps lanes n..3 to chunk 0, so the
// dummy lanes read in-bounds memory (recomputing chunk 0's CV, discarded).
//
// Body mirrors processLeavesRunAVX512 at YMM width.
// Frame: 32 bytes local (gather indices), 24 bytes args.
TEXT ·processLeavesQuadAVX512(SB), $32-24
	MOVQ	input+0(FP), BX
	MOVQ	cvs+8(FP), DI
	MOVQ	n+16(FP), AX

	// Build clamped gather index vector at SP+0: lane i = (i < n) ? i*8192 : 0.
	MOVQ	$0, 0(SP)
	MOVQ	$8192, R10; XORQ R9, R9; CMPQ AX, $1; CMOVQGT R10, R9; MOVQ R9, 8(SP)
	MOVQ	$16384, R10; XORQ R9, R9; CMPQ AX, $2; CMOVQGT R10, R9; MOVQ R9, 16(SP)
	MOVQ	$24576, R10; XORQ R9, R9; CMPQ AX, $3; CMOVQGT R10, R9; MOVQ R9, 24(SP)

	// Zero state Y0-Y24.
	ZERO_STATE_X4

	MOVQ	$48, R12

leaves_quad_avx512_loop:
	GATHER_STRIPE21_X4

	PERMUTE12_X4

	ADDQ	$168, BX
	SUBQ	$1, R12
	JNZ	leaves_quad_avx512_loop

	// Absorb final 16 lanes (128-byte remainder).
	GATHER_FINAL16_X4

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20.
	VPBROADCASTQ_IMM_0x0B_X4(Y25)
	VPXORQ	Y25, Y16, Y16
	VPBROADCASTQ_IMM_0x80_HIGH_X4(Y25)
	VPXORQ	Y25, Y20, Y20

	// Final permutation.
	PERMUTE12_X4

	// Extract CVs via VPSCATTERQQ. All 4 lanes are scattered into the cvs
	// buffer; the caller reads only the first n.
	MOVQ	$0, 0(SP)
	MOVQ	$32, 8(SP)
	MOVQ	$64, 16(SP)
	MOVQ	$96, 24(SP)
	VMOVDQU64	0(SP), Y28

	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y0, K1, 0(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y1, K1, 8(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y2, K1, 16(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y3, K1, 24(DI)(Y28*1)

	VZEROUPPER
	RET

// func processLeavesQuadPartialAVX512(input *byte, cvs *byte, n, nShared uint64, lane1 *uint64)
//
// Tail-lane variant of processLeavesQuadAVX512: processes n (1..3) contiguous
// complete chunks in lanes 0..n-1 plus a trailing partial leaf in lane n,
// whose data starts at input+n*8192 and participates for its nShared whole
// 168-byte stripes. After those stripes lane n's 25-lane state is written to
// lane1 for the Go caller to finish, the lane is re-clamped to chunk 0 as a
// dummy, and the complete lanes run their remaining stripes and padded final
// block. CVs for all 4 lanes are scattered into cvs; the caller reads the
// first n.
//
// Reads exactly n*8192 complete-chunk bytes and nShared*168 tail bytes.
// nShared must be in [0, 48].
TEXT ·processLeavesQuadPartialAVX512(SB), $32-40
	MOVQ	input+0(FP), BX
	MOVQ	cvs+8(FP), DI
	MOVQ	n+16(FP), AX
	MOVQ	nShared+24(FP), R13
	MOVQ	lane1+32(FP), SI

	// Build the gather index vector at SP+0 with n+1 active lanes:
	// lane i = (i <= n) ? i*8192 : 0.
	MOVQ	$0, 0(SP)
	MOVQ	$8192, R10; XORQ R9, R9; CMPQ AX, $1; CMOVQGE R10, R9; MOVQ R9, 8(SP)
	MOVQ	$16384, R10; XORQ R9, R9; CMPQ AX, $2; CMOVQGE R10, R9; MOVQ R9, 16(SP)
	MOVQ	$24576, R10; XORQ R9, R9; CMPQ AX, $3; CMOVQGE R10, R9; MOVQ R9, 24(SP)

	// Zero state Y0-Y24.
	ZERO_STATE_X4

	MOVQ	$48, R12
	SUBQ	R13, R12	// stripes remaining after the shared pass

	TESTQ	R13, R13
	JZ	quad_partial_export

quad_partial_shared_loop:
	GATHER_STRIPE21_X4

	PERMUTE12_X4

	ADDQ	$168, BX
	SUBQ	$1, R13
	JNZ	quad_partial_shared_loop

quad_partial_export:
	// Export the tail lane's state (qword n of Y0-Y24) for the Go tail
	// finish; the remaining stripes only affect the complete lanes.
	MOVQ	AX, CX
	MOVL	$1, R9
	SHLL	CX, R9
	KMOVB	R9, K2
	VPCOMPRESSQ	Y0, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 0(SI)
	VPCOMPRESSQ	Y1, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 8(SI)
	VPCOMPRESSQ	Y2, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 16(SI)
	VPCOMPRESSQ	Y3, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 24(SI)
	VPCOMPRESSQ	Y4, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 32(SI)
	VPCOMPRESSQ	Y5, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 40(SI)
	VPCOMPRESSQ	Y6, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 48(SI)
	VPCOMPRESSQ	Y7, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 56(SI)
	VPCOMPRESSQ	Y8, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 64(SI)
	VPCOMPRESSQ	Y9, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 72(SI)
	VPCOMPRESSQ	Y10, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 80(SI)
	VPCOMPRESSQ	Y11, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 88(SI)
	VPCOMPRESSQ	Y12, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 96(SI)
	VPCOMPRESSQ	Y13, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 104(SI)
	VPCOMPRESSQ	Y14, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 112(SI)
	VPCOMPRESSQ	Y15, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 120(SI)
	VPCOMPRESSQ	Y16, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 128(SI)
	VPCOMPRESSQ	Y17, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 136(SI)
	VPCOMPRESSQ	Y18, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 144(SI)
	VPCOMPRESSQ	Y19, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 152(SI)
	VPCOMPRESSQ	Y20, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 160(SI)
	VPCOMPRESSQ	Y21, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 168(SI)
	VPCOMPRESSQ	Y22, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 176(SI)
	VPCOMPRESSQ	Y23, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 184(SI)
	VPCOMPRESSQ	Y24, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 192(SI)

	// Re-clamp the tail lane to chunk 0: it is dead from here on but must
	// keep gathering in-bounds memory.
	MOVQ	$0, 0(SP)(AX*8)

	TESTQ	R12, R12
	JZ	quad_partial_final

quad_partial_rest_loop:
	GATHER_STRIPE21_X4

	PERMUTE12_X4

	ADDQ	$168, BX
	SUBQ	$1, R12
	JNZ	quad_partial_rest_loop

quad_partial_final:
	// Absorb final 16 lanes (128-byte remainder).
	GATHER_FINAL16_X4

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20.
	VPBROADCASTQ_IMM_0x0B_X4(Y25)
	VPXORQ	Y25, Y16, Y16
	VPBROADCASTQ_IMM_0x80_HIGH_X4(Y25)
	VPXORQ	Y25, Y20, Y20

	// Final permutation.
	PERMUTE12_X4

	// Extract CVs via VPSCATTERQQ; the caller reads only the first n.
	MOVQ	$0, 0(SP)
	MOVQ	$32, 8(SP)
	MOVQ	$64, 16(SP)
	MOVQ	$96, 24(SP)
	VMOVDQU64	0(SP), Y28

	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y0, K1, 0(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y1, K1, 8(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y2, K1, 16(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y3, K1, 24(DI)(Y28*1)

	VZEROUPPER
	RET

// func processS0LeavesQuadAVX512(input *byte, state *uint64, cvs *byte, n uint64)
//
// Fused S_0 + leaf absorption at quad width: processes n (2..4) contiguous
// 8192-byte chunks in one 4-wide pass, the final node absorbing S_0 || kt12
// marker in lane 0 and leaves 1..n-1 in the remaining lanes. As in
// processS0LeavesAVX512, the final node's last block XORs the marker word
// 0x03 at lane 16, takes no pad10*1, and is NOT put through the closing
// permutation — its 25-lane state is extracted to state first (position 136
// is set by the Go wrapper). Leaf CVs land in cvs slots 1..n-1; slot 0 and
// dummy slots hold garbage the caller ignores.
TEXT ·processS0LeavesQuadAVX512(SB), $32-32
	MOVQ	input+0(FP), BX
	MOVQ	state+8(FP), SI
	MOVQ	cvs+16(FP), DI
	MOVQ	n+24(FP), AX

	// Build clamped gather index vector at SP+0: lane i = (i < n) ? i*8192 : 0.
	MOVQ	$0, 0(SP)
	MOVQ	$8192, R10; XORQ R9, R9; CMPQ AX, $1; CMOVQGT R10, R9; MOVQ R9, 8(SP)
	MOVQ	$16384, R10; XORQ R9, R9; CMPQ AX, $2; CMOVQGT R10, R9; MOVQ R9, 16(SP)
	MOVQ	$24576, R10; XORQ R9, R9; CMPQ AX, $3; CMOVQGT R10, R9; MOVQ R9, 24(SP)

	// Zero state Y0-Y24.
	ZERO_STATE_X4

	MOVQ	$48, R12

s0quad_avx512_loop:
	GATHER_STRIPE21_X4

	PERMUTE12_X4

	ADDQ	$168, BX
	SUBQ	$1, R12
	JNZ	s0quad_avx512_loop

	// Absorb final 16 lanes (128-byte remainder).
	GATHER_FINAL16_X4

	// Lane 16: DS 0x0B for the leaves, then fix lane 0 to the kt12 marker
	// word 0x03 by XORing 0x0B^0x03 = 0x08 into element 0 only.
	VPBROADCASTQ_IMM_0x0B_X4(Y25)
	VPXORQ	Y25, Y16, Y16
	MOVQ	$0x08, AX
	VMOVQ	AX, X26
	VPXORQ	Y26, Y16, Y16

	// Lane 20: pad10*1 end 0x80 for the leaves, then cancel it in lane 0.
	VPBROADCASTQ_IMM_0x80_HIGH_X4(Y25)
	VPXORQ	Y25, Y20, Y20
	MOVQ	$0x8000000000000000, AX
	VMOVQ	AX, X26
	VPXORQ	Y26, Y20, Y20

	// Extract the final-node state (element 0 of each lane) before the
	// leaves' closing permutation scrambles it.
	VMOVQ	X0, 0(SI)
	VMOVQ	X1, 8(SI)
	VMOVQ	X2, 16(SI)
	VMOVQ	X3, 24(SI)
	VMOVQ	X4, 32(SI)
	VMOVQ	X5, 40(SI)
	VMOVQ	X6, 48(SI)
	VMOVQ	X7, 56(SI)
	VMOVQ	X8, 64(SI)
	VMOVQ	X9, 72(SI)
	VMOVQ	X10, 80(SI)
	VMOVQ	X11, 88(SI)
	VMOVQ	X12, 96(SI)
	VMOVQ	X13, 104(SI)
	VMOVQ	X14, 112(SI)
	VMOVQ	X15, 120(SI)
	VMOVQ	X16, 128(SI)
	VMOVQ	X17, 136(SI)
	VMOVQ	X18, 144(SI)
	VMOVQ	X19, 152(SI)
	VMOVQ	X20, 160(SI)
	VMOVQ	X21, 168(SI)
	VMOVQ	X22, 176(SI)
	VMOVQ	X23, 184(SI)
	VMOVQ	X24, 192(SI)

	// Closing permutation for the leaf lanes (lane 0 becomes garbage).
	PERMUTE12_X4

	// Extract CVs via VPSCATTERQQ; the caller reads only slots 1..n-1.
	MOVQ	$0, 0(SP)
	MOVQ	$32, 8(SP)
	MOVQ	$64, 16(SP)
	MOVQ	$96, 24(SP)
	VMOVDQU64	0(SP), Y28

	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y0, K1, 0(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y1, K1, 8(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y2, K1, 16(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y3, K1, 24(DI)(Y28*1)

	VZEROUPPER
	RET

// func processS0LeavesQuadTailAVX512(input *byte, state *uint64, cvs *byte, n, nShared uint64, tail *uint64)
//
// Tail-lane variant of processS0LeavesQuadAVX512: fuses S_0 (lane 0), n-1
// complete leaves (lanes 1..n-1), and the trailing partial leaf (lane n),
// which participates for its nShared whole 168-byte stripes starting at
// input+n*8192. After those stripes lane n's 25-lane state is written to
// tail for the Go caller to continue, the lane is re-clamped to chunk 0 as
// a dummy, and the remaining stripes and last block proceed as in
// processS0LeavesQuadAVX512. Leaf CVs land in cvs slots 1..n-1.
//
// Reads exactly n*8192 chunk bytes and nShared*168 tail bytes. n must be
// in [2, 3] (lane n must be free) and nShared in [0, 48].
TEXT ·processS0LeavesQuadTailAVX512(SB), $32-48
	MOVQ	input+0(FP), BX
	MOVQ	state+8(FP), SI
	MOVQ	cvs+16(FP), DI
	MOVQ	n+24(FP), AX
	MOVQ	nShared+32(FP), R13
	MOVQ	tail+40(FP), R8

	// Build the gather index vector at SP+0 with n+1 active lanes:
	// lane i = (i <= n) ? i*8192 : 0. Lane 0 is S_0, lane n the partial.
	MOVQ	$0, 0(SP)
	MOVQ	$8192, R10; XORQ R9, R9; CMPQ AX, $1; CMOVQGE R10, R9; MOVQ R9, 8(SP)
	MOVQ	$16384, R10; XORQ R9, R9; CMPQ AX, $2; CMOVQGE R10, R9; MOVQ R9, 16(SP)
	MOVQ	$24576, R10; XORQ R9, R9; CMPQ AX, $3; CMOVQGE R10, R9; MOVQ R9, 24(SP)

	// Zero state Y0-Y24.
	ZERO_STATE_X4

	MOVQ	$48, R12
	SUBQ	R13, R12	// stripes remaining after the shared pass

	TESTQ	R13, R13
	JZ	s0quad_tail_export

s0quad_tail_shared_loop:
	GATHER_STRIPE21_X4

	PERMUTE12_X4

	ADDQ	$168, BX
	SUBQ	$1, R13
	JNZ	s0quad_tail_shared_loop

s0quad_tail_export:
	// Export the tail lane's state (qword n of Y0-Y24) for the Go caller;
	// the remaining stripes only affect S_0 and the complete lanes.
	MOVQ	AX, CX
	MOVL	$1, R9
	SHLL	CX, R9
	KMOVB	R9, K2
	VPCOMPRESSQ	Y0, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 0(R8)
	VPCOMPRESSQ	Y1, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 8(R8)
	VPCOMPRESSQ	Y2, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 16(R8)
	VPCOMPRESSQ	Y3, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 24(R8)
	VPCOMPRESSQ	Y4, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 32(R8)
	VPCOMPRESSQ	Y5, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 40(R8)
	VPCOMPRESSQ	Y6, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 48(R8)
	VPCOMPRESSQ	Y7, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 56(R8)
	VPCOMPRESSQ	Y8, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 64(R8)
	VPCOMPRESSQ	Y9, K2, Y25;  VMOVQ	X25, R9; MOVQ	R9, 72(R8)
	VPCOMPRESSQ	Y10, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 80(R8)
	VPCOMPRESSQ	Y11, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 88(R8)
	VPCOMPRESSQ	Y12, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 96(R8)
	VPCOMPRESSQ	Y13, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 104(R8)
	VPCOMPRESSQ	Y14, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 112(R8)
	VPCOMPRESSQ	Y15, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 120(R8)
	VPCOMPRESSQ	Y16, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 128(R8)
	VPCOMPRESSQ	Y17, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 136(R8)
	VPCOMPRESSQ	Y18, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 144(R8)
	VPCOMPRESSQ	Y19, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 152(R8)
	VPCOMPRESSQ	Y20, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 160(R8)
	VPCOMPRESSQ	Y21, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 168(R8)
	VPCOMPRESSQ	Y22, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 176(R8)
	VPCOMPRESSQ	Y23, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 184(R8)
	VPCOMPRESSQ	Y24, K2, Y25; VMOVQ	X25, R9; MOVQ	R9, 192(R8)

	// Re-clamp the tail lane to chunk 0: it is dead from here on but must
	// keep gathering in-bounds memory.
	MOVQ	$0, 0(SP)(AX*8)

	TESTQ	R12, R12
	JZ	s0quad_tail_final

s0quad_tail_rest_loop:
	GATHER_STRIPE21_X4

	PERMUTE12_X4

	ADDQ	$168, BX
	SUBQ	$1, R12
	JNZ	s0quad_tail_rest_loop

s0quad_tail_final:
	// Absorb final 16 lanes (128-byte remainder).
	GATHER_FINAL16_X4

	// Lane 16: DS 0x0B for the leaves, then fix lane 0 to the kt12 marker
	// word 0x03 by XORing 0x0B^0x03 = 0x08 into element 0 only.
	VPBROADCASTQ_IMM_0x0B_X4(Y25)
	VPXORQ	Y25, Y16, Y16
	MOVQ	$0x08, AX
	VMOVQ	AX, X26
	VPXORQ	Y26, Y16, Y16

	// Lane 20: pad10*1 end 0x80 for the leaves, then cancel it in lane 0.
	VPBROADCASTQ_IMM_0x80_HIGH_X4(Y25)
	VPXORQ	Y25, Y20, Y20
	MOVQ	$0x8000000000000000, AX
	VMOVQ	AX, X26
	VPXORQ	Y26, Y20, Y20

	// Extract the final-node state (element 0 of each lane) before the
	// leaves' closing permutation scrambles it.
	VMOVQ	X0, 0(SI)
	VMOVQ	X1, 8(SI)
	VMOVQ	X2, 16(SI)
	VMOVQ	X3, 24(SI)
	VMOVQ	X4, 32(SI)
	VMOVQ	X5, 40(SI)
	VMOVQ	X6, 48(SI)
	VMOVQ	X7, 56(SI)
	VMOVQ	X8, 64(SI)
	VMOVQ	X9, 72(SI)
	VMOVQ	X10, 80(SI)
	VMOVQ	X11, 88(SI)
	VMOVQ	X12, 96(SI)
	VMOVQ	X13, 104(SI)
	VMOVQ	X14, 112(SI)
	VMOVQ	X15, 120(SI)
	VMOVQ	X16, 128(SI)
	VMOVQ	X17, 136(SI)
	VMOVQ	X18, 144(SI)
	VMOVQ	X19, 152(SI)
	VMOVQ	X20, 160(SI)
	VMOVQ	X21, 168(SI)
	VMOVQ	X22, 176(SI)
	VMOVQ	X23, 184(SI)
	VMOVQ	X24, 192(SI)

	// Closing permutation for the leaf lanes (lane 0 becomes garbage).
	PERMUTE12_X4

	// Extract CVs via VPSCATTERQQ; the caller reads only slots 1..n-1.
	MOVQ	$0, 0(SP)
	MOVQ	$32, 8(SP)
	MOVQ	$64, 16(SP)
	MOVQ	$96, 24(SP)
	VMOVDQU64	0(SP), Y28

	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y0, K1, 0(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y1, K1, 8(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y2, K1, 16(DI)(Y28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Y3, K1, 24(DI)(Y28*1)

	VZEROUPPER
	RET

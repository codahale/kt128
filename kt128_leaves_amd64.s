// Fused KT128 leaf processing — AVX-512 and AVX2 implementations.
//
// Each function processes 8 × 8192-byte chunks in a single call,
// producing 8 × 32-byte chain values without materializing intermediate state.

//go:build !purego

#include "textflag.h"
#include "permute_amd64_avx2.h"
#include "permute_amd64_avx512.h"

// ABSORB_LANE_X8_GATHER gathers one uint64 from 8 instances at the given byte
// offset from BX (data base pointer) using Z28 as the index vector
// ({0, stride, 2*stride, ..., 7*stride}), and XORs the result into Zlane.
// K1 is reset to all-ones before each gather.
#define ABSORB_LANE_X8_GATHER(offset, Zlane) \
	KXNORB	K1, K1, K1; \
	VPGATHERQQ	offset(BX)(Z28*1), K1, Z25; \
	VPXORQ	Z25, Zlane, Zlane

// VPBROADCASTQ_IMM_0x0B broadcasts 0x0B to all 8 qwords in Zdst.
#define VPBROADCASTQ_IMM_0x0B(Zdst) \
	MOVQ	$0x0B, AX; \
	VPBROADCASTQ	AX, Zdst

// VPBROADCASTQ_IMM_0x80_HIGH broadcasts 0x8000000000000000 to all 8 qwords in Zdst.
#define VPBROADCASTQ_IMM_0x80_HIGH(Zdst) \
	MOVQ	$0x8000000000000000, AX; \
	VPBROADCASTQ	AX, Zdst

// ZERO_STATE_X8 zeroes the Keccak state Z0-Z24.
#define ZERO_STATE_X8 \
	VPXORQ	Z0, Z0, Z0; \
	VPXORQ	Z1, Z1, Z1; \
	VPXORQ	Z2, Z2, Z2; \
	VPXORQ	Z3, Z3, Z3; \
	VPXORQ	Z4, Z4, Z4; \
	VPXORQ	Z5, Z5, Z5; \
	VPXORQ	Z6, Z6, Z6; \
	VPXORQ	Z7, Z7, Z7; \
	VPXORQ	Z8, Z8, Z8; \
	VPXORQ	Z9, Z9, Z9; \
	VPXORQ	Z10, Z10, Z10; \
	VPXORQ	Z11, Z11, Z11; \
	VPXORQ	Z12, Z12, Z12; \
	VPXORQ	Z13, Z13, Z13; \
	VPXORQ	Z14, Z14, Z14; \
	VPXORQ	Z15, Z15, Z15; \
	VPXORQ	Z16, Z16, Z16; \
	VPXORQ	Z17, Z17, Z17; \
	VPXORQ	Z18, Z18, Z18; \
	VPXORQ	Z19, Z19, Z19; \
	VPXORQ	Z20, Z20, Z20; \
	VPXORQ	Z21, Z21, Z21; \
	VPXORQ	Z22, Z22, Z22; \
	VPXORQ	Z23, Z23, Z23; \
	VPXORQ	Z24, Z24, Z24

// GATHER_STRIPE21 reloads the gather index vector from SP+0 and absorbs one
// full 168-byte stripe (21 lanes) from all 8 instances.
#define GATHER_STRIPE21 \
	VMOVDQU64	0(SP), Z28; \
	ABSORB_LANE_X8_GATHER(0*8, Z0); \
	ABSORB_LANE_X8_GATHER(1*8, Z1); \
	ABSORB_LANE_X8_GATHER(2*8, Z2); \
	ABSORB_LANE_X8_GATHER(3*8, Z3); \
	ABSORB_LANE_X8_GATHER(4*8, Z4); \
	ABSORB_LANE_X8_GATHER(5*8, Z5); \
	ABSORB_LANE_X8_GATHER(6*8, Z6); \
	ABSORB_LANE_X8_GATHER(7*8, Z7); \
	ABSORB_LANE_X8_GATHER(8*8, Z8); \
	ABSORB_LANE_X8_GATHER(9*8, Z9); \
	ABSORB_LANE_X8_GATHER(10*8, Z10); \
	ABSORB_LANE_X8_GATHER(11*8, Z11); \
	ABSORB_LANE_X8_GATHER(12*8, Z12); \
	ABSORB_LANE_X8_GATHER(13*8, Z13); \
	ABSORB_LANE_X8_GATHER(14*8, Z14); \
	ABSORB_LANE_X8_GATHER(15*8, Z15); \
	ABSORB_LANE_X8_GATHER(16*8, Z16); \
	ABSORB_LANE_X8_GATHER(17*8, Z17); \
	ABSORB_LANE_X8_GATHER(18*8, Z18); \
	ABSORB_LANE_X8_GATHER(19*8, Z19); \
	ABSORB_LANE_X8_GATHER(20*8, Z20)

// GATHER_FINAL16 reloads the gather index vector and absorbs the final
// 128-byte partial block (16 lanes) from all 8 instances.
#define GATHER_FINAL16 \
	VMOVDQU64	0(SP), Z28; \
	ABSORB_LANE_X8_GATHER(0*8, Z0); \
	ABSORB_LANE_X8_GATHER(1*8, Z1); \
	ABSORB_LANE_X8_GATHER(2*8, Z2); \
	ABSORB_LANE_X8_GATHER(3*8, Z3); \
	ABSORB_LANE_X8_GATHER(4*8, Z4); \
	ABSORB_LANE_X8_GATHER(5*8, Z5); \
	ABSORB_LANE_X8_GATHER(6*8, Z6); \
	ABSORB_LANE_X8_GATHER(7*8, Z7); \
	ABSORB_LANE_X8_GATHER(8*8, Z8); \
	ABSORB_LANE_X8_GATHER(9*8, Z9); \
	ABSORB_LANE_X8_GATHER(10*8, Z10); \
	ABSORB_LANE_X8_GATHER(11*8, Z11); \
	ABSORB_LANE_X8_GATHER(12*8, Z12); \
	ABSORB_LANE_X8_GATHER(13*8, Z13); \
	ABSORB_LANE_X8_GATHER(14*8, Z14); \
	ABSORB_LANE_X8_GATHER(15*8, Z15)

// PERMUTE12_X8 runs the 12-round Keccak-p[1600,12] permutation on Z0-Z24.
#define PERMUTE12_X8 \
	LEAQ	kt128_round_consts_2x+192(SB), R11; \
	X8_4ROUNDS_AVX512(0, 16, 32, 48); \
	X8_4ROUNDS_AVX512(64, 80, 96, 112); \
	X8_4ROUNDS_AVX512(128, 144, 160, 176)

// func processLeavesAVX512(input *byte, cvs *byte)
//
// Processes 8 × 8192-byte chunks, writing 8 × 32-byte CVs to cvs.
// Input: 8 contiguous 8192-byte blocks (total 65536 bytes).
//
// KT128 leaf constants (hardcoded):
//   Rate = 168, DS = 0x0B, BlockSize = 8192
//   48 full 168-byte stripes per 8192-byte chunk
//   128-byte remainder = 16 lanes
//   Suffix 0x0B at lane 16, pad10*1 end 0x80 at lane 20
//
// Frame: 64 bytes local (gather indices), 16 bytes args.
// Register allocation:
//   BX   = data base pointer
//   R11  = round constants pointer
//   R12  = stripe loop counter
//   Z0-Z24  = Keccak state (persistent)
//   Z25-Z31 = scratch
//   Z28     = gather index vector
TEXT ·processLeavesAVX512(SB), $64-16
	MOVQ	input+0(FP), BX
	MOVQ	cvs+8(FP), DI

	// Build gather index vector {0, 8192, 2×8192, ..., 7×8192} at SP+0.
	MOVQ	$0, 0(SP)
	MOVQ	$8192, 8(SP)
	MOVQ	$16384, 16(SP)
	MOVQ	$24576, 24(SP)
	MOVQ	$32768, 32(SP)
	MOVQ	$40960, 40(SP)
	MOVQ	$49152, 48(SP)
	MOVQ	$57344, 56(SP)

	// Zero state Z0-Z24.
	ZERO_STATE_X8

	// Loop 48 full stripes.
	MOVQ	$48, R12

leaves_avx512_loop:
	// Reload gather index vector (Z28 is clobbered by permutation).
	VMOVDQU64	0(SP), Z28

	// Absorb 21 rate lanes via gather.
	ABSORB_LANE_X8_GATHER(0*8, Z0)
	ABSORB_LANE_X8_GATHER(1*8, Z1)
	ABSORB_LANE_X8_GATHER(2*8, Z2)
	ABSORB_LANE_X8_GATHER(3*8, Z3)
	ABSORB_LANE_X8_GATHER(4*8, Z4)
	ABSORB_LANE_X8_GATHER(5*8, Z5)
	ABSORB_LANE_X8_GATHER(6*8, Z6)
	ABSORB_LANE_X8_GATHER(7*8, Z7)
	ABSORB_LANE_X8_GATHER(8*8, Z8)
	ABSORB_LANE_X8_GATHER(9*8, Z9)
	ABSORB_LANE_X8_GATHER(10*8, Z10)
	ABSORB_LANE_X8_GATHER(11*8, Z11)
	ABSORB_LANE_X8_GATHER(12*8, Z12)
	ABSORB_LANE_X8_GATHER(13*8, Z13)
	ABSORB_LANE_X8_GATHER(14*8, Z14)
	ABSORB_LANE_X8_GATHER(15*8, Z15)
	ABSORB_LANE_X8_GATHER(16*8, Z16)
	ABSORB_LANE_X8_GATHER(17*8, Z17)
	ABSORB_LANE_X8_GATHER(18*8, Z18)
	ABSORB_LANE_X8_GATHER(19*8, Z19)
	ABSORB_LANE_X8_GATHER(20*8, Z20)

	// Permute: 12 rounds = 3 × 4 rounds.
	PERMUTE12_X8

	// Advance data pointer by 168 bytes.
	ADDQ	$168, BX
	SUBQ	$1, R12
	JNZ	leaves_avx512_loop

	// Absorb final 16 lanes (128-byte remainder).
	GATHER_FINAL16

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20.
	VPBROADCASTQ_IMM_0x0B(Z25)
	VPXORQ	Z25, Z16, Z16
	VPBROADCASTQ_IMM_0x80_HIGH(Z25)
	VPXORQ	Z25, Z20, Z20

	// Final permutation.
	PERMUTE12_X8

	// Extract CVs via VPSCATTERQQ.
	MOVQ	$0, 0(SP)
	MOVQ	$32, 8(SP)
	MOVQ	$64, 16(SP)
	MOVQ	$96, 24(SP)
	MOVQ	$128, 32(SP)
	MOVQ	$160, 40(SP)
	MOVQ	$192, 48(SP)
	MOVQ	$224, 56(SP)
	VMOVDQU64	0(SP), Z28

	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z0, K1, 0(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z1, K1, 8(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z2, K1, 16(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z3, K1, 24(DI)(Z28*1)

	VZEROUPPER
	RET


// func processLeavesRunAVX512(input *byte, cvs *byte, n uint64)
//
// Direct-read remainder kernel: processes n (2..7) contiguous 8192-byte chunks
// in a single 8-wide pass, writing n×32-byte CVs to cvs. The gather index vector
// maps lanes 0..n-1 to their chunks and clamps lanes n..7 to chunk 0, so the
// dummy lanes read in-bounds memory (recomputing chunk 0's CV, which is
// discarded). This drains a 2..7 leaf remainder without padding into and zeroing
// a 64 KiB scratch buffer, and beats the serial x1 path for small remainders.
//
// Body mirrors processLeavesAVX512; only the gather index construction differs.
// Frame: 64 bytes local (gather indices), 24 bytes args.
TEXT ·processLeavesRunAVX512(SB), $64-24
	MOVQ	input+0(FP), BX
	MOVQ	cvs+8(FP), DI
	MOVQ	n+16(FP), AX

	// Build clamped gather index vector at SP+0: lane i = (i < n) ? i*8192 : 0.
	// Lane 0 is always chunk 0; lanes n..7 fall back to chunk 0 (in-bounds).
	MOVQ	$0, 0(SP)
	MOVQ	$8192, R10;  XORQ R9, R9; CMPQ AX, $1; CMOVQGT R10, R9; MOVQ R9, 8(SP)
	MOVQ	$16384, R10; XORQ R9, R9; CMPQ AX, $2; CMOVQGT R10, R9; MOVQ R9, 16(SP)
	MOVQ	$24576, R10; XORQ R9, R9; CMPQ AX, $3; CMOVQGT R10, R9; MOVQ R9, 24(SP)
	MOVQ	$32768, R10; XORQ R9, R9; CMPQ AX, $4; CMOVQGT R10, R9; MOVQ R9, 32(SP)
	MOVQ	$40960, R10; XORQ R9, R9; CMPQ AX, $5; CMOVQGT R10, R9; MOVQ R9, 40(SP)
	MOVQ	$49152, R10; XORQ R9, R9; CMPQ AX, $6; CMOVQGT R10, R9; MOVQ R9, 48(SP)
	MOVQ	$57344, R10; XORQ R9, R9; CMPQ AX, $7; CMOVQGT R10, R9; MOVQ R9, 56(SP)

	// Zero state Z0-Z24.
	ZERO_STATE_X8

	MOVQ	$48, R12

leaves_run_avx512_loop:
	GATHER_STRIPE21

	PERMUTE12_X8

	ADDQ	$168, BX
	SUBQ	$1, R12
	JNZ	leaves_run_avx512_loop

	// Absorb final 16 lanes (128-byte remainder).
	GATHER_FINAL16

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20.
	VPBROADCASTQ_IMM_0x0B(Z25)
	VPXORQ	Z25, Z16, Z16
	VPBROADCASTQ_IMM_0x80_HIGH(Z25)
	VPXORQ	Z25, Z20, Z20

	// Final permutation.
	PERMUTE12_X8

	// Extract CVs via VPSCATTERQQ. All 8 lanes are scattered into the 256-byte
	// cvs buffer; the caller reads only the first n.
	MOVQ	$0, 0(SP)
	MOVQ	$32, 8(SP)
	MOVQ	$64, 16(SP)
	MOVQ	$96, 24(SP)
	MOVQ	$128, 32(SP)
	MOVQ	$160, 40(SP)
	MOVQ	$192, 48(SP)
	MOVQ	$224, 56(SP)
	VMOVDQU64	0(SP), Z28

	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z0, K1, 0(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z1, K1, 8(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z2, K1, 16(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z3, K1, 24(DI)(Z28*1)

	VZEROUPPER
	RET


// func processLeavesRunPartialAVX512(input *byte, cvs *byte, n, nShared uint64, lane1 *uint64)
//
// Tail-lane variant of processLeavesRunAVX512: processes n (1..7) contiguous
// complete chunks in lanes 0..n-1 plus a trailing partial leaf in lane n,
// whose data starts at input+n*8192 and participates for its nShared whole
// 168-byte stripes. After those stripes lane n's 25-lane state is written to
// lane1 for the Go caller to finish (ragged tail, padding, and closing
// permutation), the lane is re-clamped to chunk 0 as a dummy, and the
// complete lanes run their remaining stripes and padded final block. CVs for
// all 8 lanes are scattered into cvs; the caller reads the first n.
//
// Reads exactly n*8192 complete-chunk bytes and nShared*168 tail bytes.
// nShared must be in [0, 48].
TEXT ·processLeavesRunPartialAVX512(SB), $64-40
	MOVQ	input+0(FP), BX
	MOVQ	cvs+8(FP), DI
	MOVQ	n+16(FP), AX
	MOVQ	nShared+24(FP), R13
	MOVQ	lane1+32(FP), SI

	// Build the gather index vector at SP+0 with n+1 active lanes:
	// lane i = (i <= n) ? i*8192 : 0. Lanes past the tail lane fall back to
	// chunk 0 (in-bounds, discarded).
	MOVQ	$0, 0(SP)
	MOVQ	$8192, R10;  XORQ R9, R9; CMPQ AX, $1; CMOVQGE R10, R9; MOVQ R9, 8(SP)
	MOVQ	$16384, R10; XORQ R9, R9; CMPQ AX, $2; CMOVQGE R10, R9; MOVQ R9, 16(SP)
	MOVQ	$24576, R10; XORQ R9, R9; CMPQ AX, $3; CMOVQGE R10, R9; MOVQ R9, 24(SP)
	MOVQ	$32768, R10; XORQ R9, R9; CMPQ AX, $4; CMOVQGE R10, R9; MOVQ R9, 32(SP)
	MOVQ	$40960, R10; XORQ R9, R9; CMPQ AX, $5; CMOVQGE R10, R9; MOVQ R9, 40(SP)
	MOVQ	$49152, R10; XORQ R9, R9; CMPQ AX, $6; CMOVQGE R10, R9; MOVQ R9, 48(SP)
	MOVQ	$57344, R10; XORQ R9, R9; CMPQ AX, $7; CMOVQGE R10, R9; MOVQ R9, 56(SP)

	// Zero state Z0-Z24.
	ZERO_STATE_X8

	MOVQ	$48, R12
	SUBQ	R13, R12	// stripes remaining after the shared pass

	TESTQ	R13, R13
	JZ	run_partial_export

run_partial_shared_loop:
	GATHER_STRIPE21

	PERMUTE12_X8

	ADDQ	$168, BX
	SUBQ	$1, R13
	JNZ	run_partial_shared_loop

run_partial_export:
	// Export the tail lane's state (qword n of Z0-Z24) for the Go tail
	// finish; the remaining stripes only affect the complete lanes.
	MOVQ	AX, CX
	MOVL	$1, R9
	SHLL	CX, R9
	KMOVB	R9, K2
	VPCOMPRESSQ	Z0, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 0(SI)
	VPCOMPRESSQ	Z1, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 8(SI)
	VPCOMPRESSQ	Z2, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 16(SI)
	VPCOMPRESSQ	Z3, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 24(SI)
	VPCOMPRESSQ	Z4, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 32(SI)
	VPCOMPRESSQ	Z5, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 40(SI)
	VPCOMPRESSQ	Z6, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 48(SI)
	VPCOMPRESSQ	Z7, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 56(SI)
	VPCOMPRESSQ	Z8, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 64(SI)
	VPCOMPRESSQ	Z9, K2, Z25;  VMOVQ X25, R9; MOVQ R9, 72(SI)
	VPCOMPRESSQ	Z10, K2, Z25; VMOVQ X25, R9; MOVQ R9, 80(SI)
	VPCOMPRESSQ	Z11, K2, Z25; VMOVQ X25, R9; MOVQ R9, 88(SI)
	VPCOMPRESSQ	Z12, K2, Z25; VMOVQ X25, R9; MOVQ R9, 96(SI)
	VPCOMPRESSQ	Z13, K2, Z25; VMOVQ X25, R9; MOVQ R9, 104(SI)
	VPCOMPRESSQ	Z14, K2, Z25; VMOVQ X25, R9; MOVQ R9, 112(SI)
	VPCOMPRESSQ	Z15, K2, Z25; VMOVQ X25, R9; MOVQ R9, 120(SI)
	VPCOMPRESSQ	Z16, K2, Z25; VMOVQ X25, R9; MOVQ R9, 128(SI)
	VPCOMPRESSQ	Z17, K2, Z25; VMOVQ X25, R9; MOVQ R9, 136(SI)
	VPCOMPRESSQ	Z18, K2, Z25; VMOVQ X25, R9; MOVQ R9, 144(SI)
	VPCOMPRESSQ	Z19, K2, Z25; VMOVQ X25, R9; MOVQ R9, 152(SI)
	VPCOMPRESSQ	Z20, K2, Z25; VMOVQ X25, R9; MOVQ R9, 160(SI)
	VPCOMPRESSQ	Z21, K2, Z25; VMOVQ X25, R9; MOVQ R9, 168(SI)
	VPCOMPRESSQ	Z22, K2, Z25; VMOVQ X25, R9; MOVQ R9, 176(SI)
	VPCOMPRESSQ	Z23, K2, Z25; VMOVQ X25, R9; MOVQ R9, 184(SI)
	VPCOMPRESSQ	Z24, K2, Z25; VMOVQ X25, R9; MOVQ R9, 192(SI)

	// Re-clamp the tail lane to chunk 0: it is dead from here on but must
	// keep gathering in-bounds memory.
	MOVQ	$0, 0(SP)(AX*8)

	TESTQ	R12, R12
	JZ	run_partial_final

run_partial_rest_loop:
	GATHER_STRIPE21

	PERMUTE12_X8

	ADDQ	$168, BX
	SUBQ	$1, R12
	JNZ	run_partial_rest_loop

run_partial_final:
	// Absorb final 16 lanes (128-byte remainder).
	GATHER_FINAL16

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20.
	VPBROADCASTQ_IMM_0x0B(Z25)
	VPXORQ	Z25, Z16, Z16
	VPBROADCASTQ_IMM_0x80_HIGH(Z25)
	VPXORQ	Z25, Z20, Z20

	// Final permutation.
	PERMUTE12_X8

	// Extract CVs via VPSCATTERQQ. All 8 lanes are scattered into the 256-byte
	// cvs buffer; the caller reads only the first n.
	MOVQ	$0, 0(SP)
	MOVQ	$32, 8(SP)
	MOVQ	$64, 16(SP)
	MOVQ	$96, 24(SP)
	MOVQ	$128, 32(SP)
	MOVQ	$160, 40(SP)
	MOVQ	$192, 48(SP)
	MOVQ	$224, 56(SP)
	VMOVDQU64	0(SP), Z28

	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z0, K1, 0(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z1, K1, 8(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z2, K1, 16(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z3, K1, 24(DI)(Z28*1)

	VZEROUPPER
	RET


// func processS0LeavesAVX512(input *byte, state *uint64, cvs *byte, n uint64)
//
// Fused S_0 + leaf absorption: processes n (2..8) contiguous 8192-byte chunks
// in a single 8-wide pass, with the final node absorbing S_0 || kt12 marker in
// lane 0 and leaves 1..n-1 in the remaining lanes. The final node has the same
// permutation schedule as a leaf (48 full stripes, a 128-byte remainder, one
// more lane) but differs in the last block: it XORs the marker word 0x03 at
// lane 16 instead of DS 0x0B, takes no pad10*1 at lane 20, and is NOT put
// through the closing permutation — its 25-lane state is extracted to state
// first (position 136 is set by the Go wrapper). Leaf CVs are scattered into
// the 256-byte cvs buffer at slots 1..n-1; slot 0 and dummy-lane slots hold
// garbage the caller ignores.
//
// Body mirrors processLeavesRunAVX512; only the last block differs.
// Frame: 64 bytes local (gather indices), 32 bytes args.
TEXT ·processS0LeavesAVX512(SB), $64-32
	MOVQ	input+0(FP), BX
	MOVQ	state+8(FP), SI
	MOVQ	cvs+16(FP), DI
	MOVQ	n+24(FP), AX

	// Build clamped gather index vector at SP+0: lane i = (i < n) ? i*8192 : 0.
	// Lane 0 is always chunk 0 (S_0); lanes n..7 fall back to chunk 0 (in-bounds).
	MOVQ	$0, 0(SP)
	MOVQ	$8192, R10;  XORQ R9, R9; CMPQ AX, $1; CMOVQGT R10, R9; MOVQ R9, 8(SP)
	MOVQ	$16384, R10; XORQ R9, R9; CMPQ AX, $2; CMOVQGT R10, R9; MOVQ R9, 16(SP)
	MOVQ	$24576, R10; XORQ R9, R9; CMPQ AX, $3; CMOVQGT R10, R9; MOVQ R9, 24(SP)
	MOVQ	$32768, R10; XORQ R9, R9; CMPQ AX, $4; CMOVQGT R10, R9; MOVQ R9, 32(SP)
	MOVQ	$40960, R10; XORQ R9, R9; CMPQ AX, $5; CMOVQGT R10, R9; MOVQ R9, 40(SP)
	MOVQ	$49152, R10; XORQ R9, R9; CMPQ AX, $6; CMOVQGT R10, R9; MOVQ R9, 48(SP)
	MOVQ	$57344, R10; XORQ R9, R9; CMPQ AX, $7; CMOVQGT R10, R9; MOVQ R9, 56(SP)

	// Zero state Z0-Z24.
	ZERO_STATE_X8

	MOVQ	$48, R12

s0leaves_avx512_loop:
	GATHER_STRIPE21

	PERMUTE12_X8

	ADDQ	$168, BX
	SUBQ	$1, R12
	JNZ	s0leaves_avx512_loop

	// Absorb final 16 lanes (128-byte remainder).
	GATHER_FINAL16

	// Lane 16: DS 0x0B for the leaves, then fix lane 0 to the kt12 marker
	// word 0x03 by XORing 0x0B^0x03 = 0x08 into element 0 only.
	VPBROADCASTQ_IMM_0x0B(Z25)
	VPXORQ	Z25, Z16, Z16
	MOVQ	$0x08, AX
	VMOVQ	AX, X26
	VPXORQ	Z26, Z16, Z16

	// Lane 20: pad10*1 end 0x80 for the leaves, then cancel it in lane 0.
	VPBROADCASTQ_IMM_0x80_HIGH(Z25)
	VPXORQ	Z25, Z20, Z20
	MOVQ	$0x8000000000000000, AX
	VMOVQ	AX, X26
	VPXORQ	Z26, Z20, Z20

	// Extract the final-node state (element 0 of each lane) before the leaves'
	// closing permutation scrambles it.
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
	PERMUTE12_X8

	// Extract CVs via VPSCATTERQQ. All 8 lanes are scattered into the 256-byte
	// cvs buffer; the caller reads only slots 1..n-1.
	MOVQ	$0, 0(SP)
	MOVQ	$32, 8(SP)
	MOVQ	$64, 16(SP)
	MOVQ	$96, 24(SP)
	MOVQ	$128, 32(SP)
	MOVQ	$160, 40(SP)
	MOVQ	$192, 48(SP)
	MOVQ	$224, 56(SP)
	VMOVDQU64	0(SP), Z28

	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z0, K1, 0(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z1, K1, 8(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z2, K1, 16(DI)(Z28*1)
	KXNORB	K1, K1, K1
	VPSCATTERQQ	Z3, K1, 24(DI)(Z28*1)

	VZEROUPPER
	RET


// ABSORB_LANE_X4 XORs lane i from 4 input pointers into the state buffer.
// AX=in0, CX=in1, DX=in2, BX=in3, R8=state buffer.
#define ABSORB_LANE_X4(i) \
	MOVQ	i*8(AX), R10; XORQ	R10, i*32+0(R8); \
	MOVQ	i*8(CX), R10; XORQ	R10, i*32+8(R8); \
	MOVQ	i*8(DX), R10; XORQ	R10, i*32+16(R8); \
	MOVQ	i*8(BX), R10; XORQ	R10, i*32+24(R8)

// func processLeavesAVX2(input *byte, cvs *byte)
//
// Processes 8 × 8192-byte chunks via 2× x4 AVX2, writing 8 × 32-byte CVs.
//
// Frame: 1648 bytes local (0-799 = buffer A, 800-1599 = buffer B,
//        1600-1647 = 4 input ptrs + count + saved output ptr), 16 bytes args.
TEXT ·processLeavesAVX2(SB), $1648-16
	MOVQ	input+0(FP), AX
	MOVQ	cvs+8(FP), DI
	MOVQ	DI, 1640(SP)		// save output pointer

	// === Half 0: instances 0-3 ===
	// Input pointers: in + {0, 1, 2, 3}×8192.
	MOVQ	AX, 1600(SP)		// in0
	LEAQ	8192(AX), R10
	MOVQ	R10, 1608(SP)		// in1
	LEAQ	8192(R10), R10
	MOVQ	R10, 1616(SP)		// in2
	LEAQ	8192(R10), R10
	MOVQ	R10, 1624(SP)		// in3
	MOVQ	$48, 1632(SP)		// stripe count

	// Zero buffer A (25 lanes × 32 bytes = 800 bytes).
	VPXOR	Y0, Y0, Y0
	VMOVDQU	Y0, 0*32(SP)
	VMOVDQU	Y0, 1*32(SP)
	VMOVDQU	Y0, 2*32(SP)
	VMOVDQU	Y0, 3*32(SP)
	VMOVDQU	Y0, 4*32(SP)
	VMOVDQU	Y0, 5*32(SP)
	VMOVDQU	Y0, 6*32(SP)
	VMOVDQU	Y0, 7*32(SP)
	VMOVDQU	Y0, 8*32(SP)
	VMOVDQU	Y0, 9*32(SP)
	VMOVDQU	Y0, 10*32(SP)
	VMOVDQU	Y0, 11*32(SP)
	VMOVDQU	Y0, 12*32(SP)
	VMOVDQU	Y0, 13*32(SP)
	VMOVDQU	Y0, 14*32(SP)
	VMOVDQU	Y0, 15*32(SP)
	VMOVDQU	Y0, 16*32(SP)
	VMOVDQU	Y0, 17*32(SP)
	VMOVDQU	Y0, 18*32(SP)
	VMOVDQU	Y0, 19*32(SP)
	VMOVDQU	Y0, 20*32(SP)
	VMOVDQU	Y0, 21*32(SP)
	VMOVDQU	Y0, 22*32(SP)
	VMOVDQU	Y0, 23*32(SP)
	VMOVDQU	Y0, 24*32(SP)

leaves_avx2_loop_a:
	CMPQ	1632(SP), $0
	JEQ	leaves_avx2_final_a

	// Absorb from instances 0-3.
	LEAQ	0(SP), R8
	MOVQ	1600(SP), AX
	MOVQ	1608(SP), CX
	MOVQ	1616(SP), DX
	MOVQ	1624(SP), BX

	ABSORB_LANE_X4(0)
	ABSORB_LANE_X4(1)
	ABSORB_LANE_X4(2)
	ABSORB_LANE_X4(3)
	ABSORB_LANE_X4(4)
	ABSORB_LANE_X4(5)
	ABSORB_LANE_X4(6)
	ABSORB_LANE_X4(7)
	ABSORB_LANE_X4(8)
	ABSORB_LANE_X4(9)
	ABSORB_LANE_X4(10)
	ABSORB_LANE_X4(11)
	ABSORB_LANE_X4(12)
	ABSORB_LANE_X4(13)
	ABSORB_LANE_X4(14)
	ABSORB_LANE_X4(15)
	ABSORB_LANE_X4(16)
	ABSORB_LANE_X4(17)
	ABSORB_LANE_X4(18)
	ABSORB_LANE_X4(19)
	ABSORB_LANE_X4(20)

	ADDQ	$168, AX
	ADDQ	$168, CX
	ADDQ	$168, DX
	ADDQ	$168, BX
	MOVQ	AX, 1600(SP)
	MOVQ	CX, 1608(SP)
	MOVQ	DX, 1616(SP)
	MOVQ	BX, 1624(SP)
	SUBQ	$1, 1632(SP)

	// Permute.
	LEAQ	0(SP), R8
	LEAQ	800(SP), R9
	LEAQ	kt128_round_consts_4x+384(SB), R11
	MOVQ	$12, R10

	PCALIGN	$16
leaves_avx2_round_a:
	X4_KECCAK_ROUND
	XCHGQ	R8, R9
	ADDQ	$32, R11
	SUBQ	$1, R10
	JNZ	leaves_avx2_round_a

	JMP	leaves_avx2_loop_a

leaves_avx2_final_a:
	// Absorb final 16 lanes from instances 0-3.
	LEAQ	0(SP), R8
	MOVQ	1600(SP), AX
	MOVQ	1608(SP), CX
	MOVQ	1616(SP), DX
	MOVQ	1624(SP), BX

	ABSORB_LANE_X4(0)
	ABSORB_LANE_X4(1)
	ABSORB_LANE_X4(2)
	ABSORB_LANE_X4(3)
	ABSORB_LANE_X4(4)
	ABSORB_LANE_X4(5)
	ABSORB_LANE_X4(6)
	ABSORB_LANE_X4(7)
	ABSORB_LANE_X4(8)
	ABSORB_LANE_X4(9)
	ABSORB_LANE_X4(10)
	ABSORB_LANE_X4(11)
	ABSORB_LANE_X4(12)
	ABSORB_LANE_X4(13)
	ABSORB_LANE_X4(14)
	ABSORB_LANE_X4(15)

	// XOR padding: DS=0x0B into lane 16 (all 4 instances), pad end into lane 20.
	LEAQ	0(SP), R8
	MOVQ	$0x0B, R10
	XORQ	R10, 16*32+0(R8)
	XORQ	R10, 16*32+8(R8)
	XORQ	R10, 16*32+16(R8)
	XORQ	R10, 16*32+24(R8)
	MOVQ	$0x8000000000000000, R10
	XORQ	R10, 20*32+0(R8)
	XORQ	R10, 20*32+8(R8)
	XORQ	R10, 20*32+16(R8)
	XORQ	R10, 20*32+24(R8)

	// Final permutation.
	LEAQ	0(SP), R8
	LEAQ	800(SP), R9
	LEAQ	kt128_round_consts_4x+384(SB), R11
	MOVQ	$12, R10

	PCALIGN	$16
leaves_avx2_final_round_a:
	X4_KECCAK_ROUND
	XCHGQ	R8, R9
	ADDQ	$32, R11
	SUBQ	$1, R10
	JNZ	leaves_avx2_final_round_a

	// Extract CVs for instances 0-3 via UNPACK+PERM128 transpose.
	MOVQ	1640(SP), DI
	LEAQ	0(SP), R8

	VMOVDQU	0*32(R8), Y0	// lane 0: {i0, i1, i2, i3}
	VMOVDQU	1*32(R8), Y1	// lane 1
	VMOVDQU	2*32(R8), Y2	// lane 2
	VMOVDQU	3*32(R8), Y3	// lane 3

	VPUNPCKLQDQ	Y1, Y0, Y4	// {i0_l0, i0_l1, i2_l0, i2_l1}
	VPUNPCKHQDQ	Y1, Y0, Y5	// {i1_l0, i1_l1, i3_l0, i3_l1}
	VPUNPCKLQDQ	Y3, Y2, Y6	// {i0_l2, i0_l3, i2_l2, i2_l3}
	VPUNPCKHQDQ	Y3, Y2, Y7	// {i1_l2, i1_l3, i3_l2, i3_l3}

	VPERM2F128	$0x20, Y6, Y4, Y0	// inst0 CV
	VPERM2F128	$0x20, Y7, Y5, Y1	// inst1 CV
	VPERM2F128	$0x31, Y6, Y4, Y2	// inst2 CV
	VPERM2F128	$0x31, Y7, Y5, Y3	// inst3 CV

	VMOVDQU	Y0, 0*32(DI)
	VMOVDQU	Y1, 1*32(DI)
	VMOVDQU	Y2, 2*32(DI)
	VMOVDQU	Y3, 3*32(DI)

	// === Half 1: instances 4-7 ===
	MOVQ	input+0(FP), AX
	LEAQ	32768(AX), AX		// in + 4×8192
	MOVQ	AX, 1600(SP)		// in4
	LEAQ	8192(AX), R10
	MOVQ	R10, 1608(SP)		// in5
	LEAQ	8192(R10), R10
	MOVQ	R10, 1616(SP)		// in6
	LEAQ	8192(R10), R10
	MOVQ	R10, 1624(SP)		// in7
	MOVQ	$48, 1632(SP)		// stripe count

	// Zero buffer A.
	VPXOR	Y0, Y0, Y0
	VMOVDQU	Y0, 0*32(SP)
	VMOVDQU	Y0, 1*32(SP)
	VMOVDQU	Y0, 2*32(SP)
	VMOVDQU	Y0, 3*32(SP)
	VMOVDQU	Y0, 4*32(SP)
	VMOVDQU	Y0, 5*32(SP)
	VMOVDQU	Y0, 6*32(SP)
	VMOVDQU	Y0, 7*32(SP)
	VMOVDQU	Y0, 8*32(SP)
	VMOVDQU	Y0, 9*32(SP)
	VMOVDQU	Y0, 10*32(SP)
	VMOVDQU	Y0, 11*32(SP)
	VMOVDQU	Y0, 12*32(SP)
	VMOVDQU	Y0, 13*32(SP)
	VMOVDQU	Y0, 14*32(SP)
	VMOVDQU	Y0, 15*32(SP)
	VMOVDQU	Y0, 16*32(SP)
	VMOVDQU	Y0, 17*32(SP)
	VMOVDQU	Y0, 18*32(SP)
	VMOVDQU	Y0, 19*32(SP)
	VMOVDQU	Y0, 20*32(SP)
	VMOVDQU	Y0, 21*32(SP)
	VMOVDQU	Y0, 22*32(SP)
	VMOVDQU	Y0, 23*32(SP)
	VMOVDQU	Y0, 24*32(SP)

leaves_avx2_loop_b:
	CMPQ	1632(SP), $0
	JEQ	leaves_avx2_final_b

	LEAQ	0(SP), R8
	MOVQ	1600(SP), AX
	MOVQ	1608(SP), CX
	MOVQ	1616(SP), DX
	MOVQ	1624(SP), BX

	ABSORB_LANE_X4(0)
	ABSORB_LANE_X4(1)
	ABSORB_LANE_X4(2)
	ABSORB_LANE_X4(3)
	ABSORB_LANE_X4(4)
	ABSORB_LANE_X4(5)
	ABSORB_LANE_X4(6)
	ABSORB_LANE_X4(7)
	ABSORB_LANE_X4(8)
	ABSORB_LANE_X4(9)
	ABSORB_LANE_X4(10)
	ABSORB_LANE_X4(11)
	ABSORB_LANE_X4(12)
	ABSORB_LANE_X4(13)
	ABSORB_LANE_X4(14)
	ABSORB_LANE_X4(15)
	ABSORB_LANE_X4(16)
	ABSORB_LANE_X4(17)
	ABSORB_LANE_X4(18)
	ABSORB_LANE_X4(19)
	ABSORB_LANE_X4(20)

	ADDQ	$168, AX
	ADDQ	$168, CX
	ADDQ	$168, DX
	ADDQ	$168, BX
	MOVQ	AX, 1600(SP)
	MOVQ	CX, 1608(SP)
	MOVQ	DX, 1616(SP)
	MOVQ	BX, 1624(SP)
	SUBQ	$1, 1632(SP)

	LEAQ	0(SP), R8
	LEAQ	800(SP), R9
	LEAQ	kt128_round_consts_4x+384(SB), R11
	MOVQ	$12, R10

	PCALIGN	$16
leaves_avx2_round_b:
	X4_KECCAK_ROUND
	XCHGQ	R8, R9
	ADDQ	$32, R11
	SUBQ	$1, R10
	JNZ	leaves_avx2_round_b

	JMP	leaves_avx2_loop_b

leaves_avx2_final_b:
	// Absorb final 16 lanes from instances 4-7.
	LEAQ	0(SP), R8
	MOVQ	1600(SP), AX
	MOVQ	1608(SP), CX
	MOVQ	1616(SP), DX
	MOVQ	1624(SP), BX

	ABSORB_LANE_X4(0)
	ABSORB_LANE_X4(1)
	ABSORB_LANE_X4(2)
	ABSORB_LANE_X4(3)
	ABSORB_LANE_X4(4)
	ABSORB_LANE_X4(5)
	ABSORB_LANE_X4(6)
	ABSORB_LANE_X4(7)
	ABSORB_LANE_X4(8)
	ABSORB_LANE_X4(9)
	ABSORB_LANE_X4(10)
	ABSORB_LANE_X4(11)
	ABSORB_LANE_X4(12)
	ABSORB_LANE_X4(13)
	ABSORB_LANE_X4(14)
	ABSORB_LANE_X4(15)

	// XOR padding.
	LEAQ	0(SP), R8
	MOVQ	$0x0B, R10
	XORQ	R10, 16*32+0(R8)
	XORQ	R10, 16*32+8(R8)
	XORQ	R10, 16*32+16(R8)
	XORQ	R10, 16*32+24(R8)
	MOVQ	$0x8000000000000000, R10
	XORQ	R10, 20*32+0(R8)
	XORQ	R10, 20*32+8(R8)
	XORQ	R10, 20*32+16(R8)
	XORQ	R10, 20*32+24(R8)

	// Final permutation.
	LEAQ	0(SP), R8
	LEAQ	800(SP), R9
	LEAQ	kt128_round_consts_4x+384(SB), R11
	MOVQ	$12, R10

	PCALIGN	$16
leaves_avx2_final_round_b:
	X4_KECCAK_ROUND
	XCHGQ	R8, R9
	ADDQ	$32, R11
	SUBQ	$1, R10
	JNZ	leaves_avx2_final_round_b

	// Extract CVs for instances 4-7 via UNPACK+PERM128 transpose.
	MOVQ	1640(SP), DI
	LEAQ	0(SP), R8

	VMOVDQU	0*32(R8), Y0	// lane 0: {i4, i5, i6, i7}
	VMOVDQU	1*32(R8), Y1	// lane 1
	VMOVDQU	2*32(R8), Y2	// lane 2
	VMOVDQU	3*32(R8), Y3	// lane 3

	VPUNPCKLQDQ	Y1, Y0, Y4
	VPUNPCKHQDQ	Y1, Y0, Y5
	VPUNPCKLQDQ	Y3, Y2, Y6
	VPUNPCKHQDQ	Y3, Y2, Y7

	VPERM2F128	$0x20, Y6, Y4, Y0	// inst4 CV
	VPERM2F128	$0x20, Y7, Y5, Y1	// inst5 CV
	VPERM2F128	$0x31, Y6, Y4, Y2	// inst6 CV
	VPERM2F128	$0x31, Y7, Y5, Y3	// inst7 CV

	VMOVDQU	Y0, 4*32(DI)
	VMOVDQU	Y1, 5*32(DI)
	VMOVDQU	Y2, 6*32(DI)
	VMOVDQU	Y3, 7*32(DI)

	VZEROUPPER
	RET


// func processLeavesQuadAVX2(in0, in1, in2, in3, cvs *byte)
//
// Processes 4 chunks read from four independent pointers via one x4 AVX2 pass,
// writing 4 × 32-byte CVs to cvs. This is one half of processLeavesAVX2 factored
// out so the remainder path can drain leaves directly from the input: the caller
// points dummy lanes at an in-bounds chunk and reads only the live CVs. Pointers
// need not be contiguous, so a 2..7 leaf remainder runs as one or two x4 passes
// with no padded scratch buffer.
//
// Frame: 1648 bytes local (0-799 = buffer A, 800-1599 = buffer B, 1600-1647 =
//        4 input ptrs + count + output ptr), 40 bytes args.
TEXT ·processLeavesQuadAVX2(SB), $1648-40
	MOVQ	in0+0(FP), AX
	MOVQ	AX, 1600(SP)
	MOVQ	in1+8(FP), R10
	MOVQ	R10, 1608(SP)
	MOVQ	in2+16(FP), R10
	MOVQ	R10, 1616(SP)
	MOVQ	in3+24(FP), R10
	MOVQ	R10, 1624(SP)
	MOVQ	cvs+32(FP), R10
	MOVQ	R10, 1640(SP)
	MOVQ	$48, 1632(SP)

	// Zero buffer A (25 lanes × 32 bytes = 800 bytes).
	VPXOR	Y0, Y0, Y0
	VMOVDQU	Y0, 0*32(SP)
	VMOVDQU	Y0, 1*32(SP)
	VMOVDQU	Y0, 2*32(SP)
	VMOVDQU	Y0, 3*32(SP)
	VMOVDQU	Y0, 4*32(SP)
	VMOVDQU	Y0, 5*32(SP)
	VMOVDQU	Y0, 6*32(SP)
	VMOVDQU	Y0, 7*32(SP)
	VMOVDQU	Y0, 8*32(SP)
	VMOVDQU	Y0, 9*32(SP)
	VMOVDQU	Y0, 10*32(SP)
	VMOVDQU	Y0, 11*32(SP)
	VMOVDQU	Y0, 12*32(SP)
	VMOVDQU	Y0, 13*32(SP)
	VMOVDQU	Y0, 14*32(SP)
	VMOVDQU	Y0, 15*32(SP)
	VMOVDQU	Y0, 16*32(SP)
	VMOVDQU	Y0, 17*32(SP)
	VMOVDQU	Y0, 18*32(SP)
	VMOVDQU	Y0, 19*32(SP)
	VMOVDQU	Y0, 20*32(SP)
	VMOVDQU	Y0, 21*32(SP)
	VMOVDQU	Y0, 22*32(SP)
	VMOVDQU	Y0, 23*32(SP)
	VMOVDQU	Y0, 24*32(SP)

leaves_quad_avx2_loop:
	CMPQ	1632(SP), $0
	JEQ	leaves_quad_avx2_final

	LEAQ	0(SP), R8
	MOVQ	1600(SP), AX
	MOVQ	1608(SP), CX
	MOVQ	1616(SP), DX
	MOVQ	1624(SP), BX

	ABSORB_LANE_X4(0)
	ABSORB_LANE_X4(1)
	ABSORB_LANE_X4(2)
	ABSORB_LANE_X4(3)
	ABSORB_LANE_X4(4)
	ABSORB_LANE_X4(5)
	ABSORB_LANE_X4(6)
	ABSORB_LANE_X4(7)
	ABSORB_LANE_X4(8)
	ABSORB_LANE_X4(9)
	ABSORB_LANE_X4(10)
	ABSORB_LANE_X4(11)
	ABSORB_LANE_X4(12)
	ABSORB_LANE_X4(13)
	ABSORB_LANE_X4(14)
	ABSORB_LANE_X4(15)
	ABSORB_LANE_X4(16)
	ABSORB_LANE_X4(17)
	ABSORB_LANE_X4(18)
	ABSORB_LANE_X4(19)
	ABSORB_LANE_X4(20)

	ADDQ	$168, AX
	ADDQ	$168, CX
	ADDQ	$168, DX
	ADDQ	$168, BX
	MOVQ	AX, 1600(SP)
	MOVQ	CX, 1608(SP)
	MOVQ	DX, 1616(SP)
	MOVQ	BX, 1624(SP)
	SUBQ	$1, 1632(SP)

	LEAQ	0(SP), R8
	LEAQ	800(SP), R9
	LEAQ	kt128_round_consts_4x+384(SB), R11
	MOVQ	$12, R10

	PCALIGN	$16
leaves_quad_avx2_round:
	X4_KECCAK_ROUND
	XCHGQ	R8, R9
	ADDQ	$32, R11
	SUBQ	$1, R10
	JNZ	leaves_quad_avx2_round

	JMP	leaves_quad_avx2_loop

leaves_quad_avx2_final:
	// Absorb final 16 lanes.
	LEAQ	0(SP), R8
	MOVQ	1600(SP), AX
	MOVQ	1608(SP), CX
	MOVQ	1616(SP), DX
	MOVQ	1624(SP), BX

	ABSORB_LANE_X4(0)
	ABSORB_LANE_X4(1)
	ABSORB_LANE_X4(2)
	ABSORB_LANE_X4(3)
	ABSORB_LANE_X4(4)
	ABSORB_LANE_X4(5)
	ABSORB_LANE_X4(6)
	ABSORB_LANE_X4(7)
	ABSORB_LANE_X4(8)
	ABSORB_LANE_X4(9)
	ABSORB_LANE_X4(10)
	ABSORB_LANE_X4(11)
	ABSORB_LANE_X4(12)
	ABSORB_LANE_X4(13)
	ABSORB_LANE_X4(14)
	ABSORB_LANE_X4(15)

	// XOR padding: DS=0x0B into lane 16 (all 4 instances), pad end into lane 20.
	LEAQ	0(SP), R8
	MOVQ	$0x0B, R10
	XORQ	R10, 16*32+0(R8)
	XORQ	R10, 16*32+8(R8)
	XORQ	R10, 16*32+16(R8)
	XORQ	R10, 16*32+24(R8)
	MOVQ	$0x8000000000000000, R10
	XORQ	R10, 20*32+0(R8)
	XORQ	R10, 20*32+8(R8)
	XORQ	R10, 20*32+16(R8)
	XORQ	R10, 20*32+24(R8)

	// Final permutation.
	LEAQ	0(SP), R8
	LEAQ	800(SP), R9
	LEAQ	kt128_round_consts_4x+384(SB), R11
	MOVQ	$12, R10

	PCALIGN	$16
leaves_quad_avx2_final_round:
	X4_KECCAK_ROUND
	XCHGQ	R8, R9
	ADDQ	$32, R11
	SUBQ	$1, R10
	JNZ	leaves_quad_avx2_final_round

	// Extract 4 CVs via UNPACK+PERM128 transpose.
	MOVQ	1640(SP), DI
	LEAQ	0(SP), R8

	VMOVDQU	0*32(R8), Y0	// lane 0: {i0, i1, i2, i3}
	VMOVDQU	1*32(R8), Y1	// lane 1
	VMOVDQU	2*32(R8), Y2	// lane 2
	VMOVDQU	3*32(R8), Y3	// lane 3

	VPUNPCKLQDQ	Y1, Y0, Y4
	VPUNPCKHQDQ	Y1, Y0, Y5
	VPUNPCKLQDQ	Y3, Y2, Y6
	VPUNPCKHQDQ	Y3, Y2, Y7

	VPERM2F128	$0x20, Y6, Y4, Y0	// inst0 CV
	VPERM2F128	$0x20, Y7, Y5, Y1	// inst1 CV
	VPERM2F128	$0x31, Y6, Y4, Y2	// inst2 CV
	VPERM2F128	$0x31, Y7, Y5, Y3	// inst3 CV

	VMOVDQU	Y0, 0*32(DI)
	VMOVDQU	Y1, 1*32(DI)
	VMOVDQU	Y2, 2*32(DI)
	VMOVDQU	Y3, 3*32(DI)

	VZEROUPPER
	RET

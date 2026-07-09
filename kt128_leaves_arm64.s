// Fused KT128 leaf processing — ARM64 NEON x2 pair kernels.
//
// Each kernel processes two 8192-byte chunks packed into the 128-bit NEON
// registers, producing chain values (or the fused S_0 state) without
// materializing intermediate state. Larger batches ride the hybrid
// scalar/NEON x5 kernel (kt128_leaves_x5_arm64.s) or chains of pairs.

//go:build !purego

#include "textflag.h"
#include "permute_arm64.h"

// ABSORB_STRIPE_X2 XORs one 168-byte stripe from two input pointers (IN0, IN1)
// into state registers V0-V20 (21 rate lanes). Uses V25-V26 as temps.
// Each lane is {in0_val, in1_val} packed into a 128-bit vector.
#define ABSORB_STRIPE_X2(IN0, IN1) \
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V0.B16, V0.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V1.B16, V1.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V2.B16, V2.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V3.B16, V3.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V4.B16, V4.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V5.B16, V5.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V6.B16, V6.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V7.B16, V7.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V8.B16, V8.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V9.B16, V9.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V10.B16, V10.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V11.B16, V11.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V12.B16, V12.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V13.B16, V13.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V14.B16, V14.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V15.B16, V15.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V16.B16, V16.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V17.B16, V17.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V18.B16, V18.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V19.B16, V19.B16; \
	\
	VLD1	(IN0), [V25.D1]; ADD $8, IN0; \
	VLD1	(IN1), [V26.D1]; ADD $8, IN1; \
	VZIP1	V26.D2, V25.D2, V25.D2; \
	VEOR	V25.B16, V20.B16, V20.B16

// func processLeavesPairARM64(input *byte, cvs *byte)
//
// Processes 2 contiguous 8192-byte chunks, writing 2 × 32-byte CVs to cvs,
// reading directly from the input with no scratch buffer.
TEXT ·processLeavesPairARM64(SB), NOSPLIT, $0-16
	MOVD	input+0(FP), R0		// input base (2 chunks)
	MOVD	cvs+8(FP), R6		// output CVs base

	MOVD	R0, R2			// in0
	ADD	$8192, R0, R3		// in1

	// Zero state V0-V24.
	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16
	VEOR	V4.B16, V4.B16, V4.B16
	VEOR	V5.B16, V5.B16, V5.B16
	VEOR	V6.B16, V6.B16, V6.B16
	VEOR	V7.B16, V7.B16, V7.B16
	VEOR	V8.B16, V8.B16, V8.B16
	VEOR	V9.B16, V9.B16, V9.B16
	VEOR	V10.B16, V10.B16, V10.B16
	VEOR	V11.B16, V11.B16, V11.B16
	VEOR	V12.B16, V12.B16, V12.B16
	VEOR	V13.B16, V13.B16, V13.B16
	VEOR	V14.B16, V14.B16, V14.B16
	VEOR	V15.B16, V15.B16, V15.B16
	VEOR	V16.B16, V16.B16, V16.B16
	VEOR	V17.B16, V17.B16, V17.B16
	VEOR	V18.B16, V18.B16, V18.B16
	VEOR	V19.B16, V19.B16, V19.B16
	VEOR	V20.B16, V20.B16, V20.B16
	VEOR	V21.B16, V21.B16, V21.B16
	VEOR	V22.B16, V22.B16, V22.B16
	VEOR	V23.B16, V23.B16, V23.B16
	VEOR	V24.B16, V24.B16, V24.B16

	MOVD	$48, R4

leaves_arm64_pair_loop:
	ABSORB_STRIPE_X2(R2, R3)

	MOVD	$kt128_round_consts(SB), R1
	ADD	$96, R1
	KECCAK_12_ROUNDS

	SUBS	$1, R4
	BNE	leaves_arm64_pair_loop

	// Absorb final 16 lanes.
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V0.B16, V0.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V1.B16, V1.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V2.B16, V2.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V3.B16, V3.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V4.B16, V4.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V5.B16, V5.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V6.B16, V6.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V7.B16, V7.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V8.B16, V8.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V9.B16, V9.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V10.B16, V10.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V11.B16, V11.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V12.B16, V12.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V13.B16, V13.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V14.B16, V14.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V15.B16, V15.B16

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20.
	MOVD	$0x0B, R9
	VDUP	R9, V25.D2
	VEOR	V25.B16, V16.B16, V16.B16
	MOVD	$0x8000000000000000, R9
	VDUP	R9, V25.D2
	VEOR	V25.B16, V20.B16, V20.B16

	// Final permutation.
	MOVD	$kt128_round_consts(SB), R1
	ADD	$96, R1
	KECCAK_12_ROUNDS

	// Extract CVs for the pair.
	VST1	[V0.D1], (R6); ADD $8, R6
	VST1	[V1.D1], (R6); ADD $8, R6
	VST1	[V2.D1], (R6); ADD $8, R6
	VST1	[V3.D1], (R6); ADD $8, R6
	VDUP	V0.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V1.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V2.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V3.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6

	RET

// func processS0LeafPairARM64(input *byte, state *uint64, cv *byte)
//
// Fused S_0 + first-leaf absorption. input is 2 contiguous 8192-byte chunks:
// S_0 at +0 and the first leaf at +8192. The final node absorbing
// S_0 || kt12 marker has the same permutation schedule as a leaf (48 full
// stripes plus a 128-byte remainder and one more lane), so both ride one x2
// pair: the final node in lane d[0], the leaf in d[1]. They differ only in
// the last block: the final node XORs the marker word 0x03 at lane 16 and is
// NOT permuted (it stays mid-block at position 136, set by the Go wrapper),
// while the leaf XORs DS 0x0B at lane 16 and pad10*1 0x80 at lane 20 and is.
// The 25-lane final-node state is written to state before the leaf's closing
// permutation scrambles that lane; the leaf's 32-byte CV is written to cv.
TEXT ·processS0LeafPairARM64(SB), NOSPLIT, $0-24
	MOVD	input+0(FP), R0		// input base (2 chunks)
	MOVD	state+8(FP), R7		// output final-node state base
	MOVD	cv+16(FP), R6		// output CV base

	MOVD	R0, R2			// in0 = S_0
	ADD	$8192, R0, R3		// in1 = leaf

	// Zero state V0-V24.
	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16
	VEOR	V4.B16, V4.B16, V4.B16
	VEOR	V5.B16, V5.B16, V5.B16
	VEOR	V6.B16, V6.B16, V6.B16
	VEOR	V7.B16, V7.B16, V7.B16
	VEOR	V8.B16, V8.B16, V8.B16
	VEOR	V9.B16, V9.B16, V9.B16
	VEOR	V10.B16, V10.B16, V10.B16
	VEOR	V11.B16, V11.B16, V11.B16
	VEOR	V12.B16, V12.B16, V12.B16
	VEOR	V13.B16, V13.B16, V13.B16
	VEOR	V14.B16, V14.B16, V14.B16
	VEOR	V15.B16, V15.B16, V15.B16
	VEOR	V16.B16, V16.B16, V16.B16
	VEOR	V17.B16, V17.B16, V17.B16
	VEOR	V18.B16, V18.B16, V18.B16
	VEOR	V19.B16, V19.B16, V19.B16
	VEOR	V20.B16, V20.B16, V20.B16
	VEOR	V21.B16, V21.B16, V21.B16
	VEOR	V22.B16, V22.B16, V22.B16
	VEOR	V23.B16, V23.B16, V23.B16
	VEOR	V24.B16, V24.B16, V24.B16

	MOVD	$48, R4

s0leaf_arm64_pair_loop:
	ABSORB_STRIPE_X2(R2, R3)

	MOVD	$kt128_round_consts(SB), R1
	ADD	$96, R1
	KECCAK_12_ROUNDS

	SUBS	$1, R4
	BNE	s0leaf_arm64_pair_loop

	// Absorb final 16 lanes.
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V0.B16, V0.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V1.B16, V1.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V2.B16, V2.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V3.B16, V3.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V4.B16, V4.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V5.B16, V5.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V6.B16, V6.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V7.B16, V7.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V8.B16, V8.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V9.B16, V9.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V10.B16, V10.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V11.B16, V11.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V12.B16, V12.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V13.B16, V13.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V14.B16, V14.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VLD1 (R3), [V26.D1]; ADD $8, R3; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, V15.B16, V15.B16

	// Lane 16: kt12 marker word 0x03 for the final node (d[0]), leaf DS 0x0B
	// for the leaf (d[1]).
	MOVD	$0x0B, R9
	VDUP	R9, V25.D2
	MOVD	$0x03, R9
	VMOV	R9, V25.D[0]
	VEOR	V25.B16, V16.B16, V16.B16

	// Lane 20: pad10*1 end 0x80 for the leaf only.
	VEOR	V25.B16, V25.B16, V25.B16
	MOVD	$0x8000000000000000, R9
	VMOV	R9, V25.D[1]
	VEOR	V25.B16, V20.B16, V20.B16

	// Extract the final-node state (d[0] of all 25 lanes) before the leaf's
	// closing permutation scrambles it.
	VST1	[V0.D1], (R7); ADD $8, R7
	VST1	[V1.D1], (R7); ADD $8, R7
	VST1	[V2.D1], (R7); ADD $8, R7
	VST1	[V3.D1], (R7); ADD $8, R7
	VST1	[V4.D1], (R7); ADD $8, R7
	VST1	[V5.D1], (R7); ADD $8, R7
	VST1	[V6.D1], (R7); ADD $8, R7
	VST1	[V7.D1], (R7); ADD $8, R7
	VST1	[V8.D1], (R7); ADD $8, R7
	VST1	[V9.D1], (R7); ADD $8, R7
	VST1	[V10.D1], (R7); ADD $8, R7
	VST1	[V11.D1], (R7); ADD $8, R7
	VST1	[V12.D1], (R7); ADD $8, R7
	VST1	[V13.D1], (R7); ADD $8, R7
	VST1	[V14.D1], (R7); ADD $8, R7
	VST1	[V15.D1], (R7); ADD $8, R7
	VST1	[V16.D1], (R7); ADD $8, R7
	VST1	[V17.D1], (R7); ADD $8, R7
	VST1	[V18.D1], (R7); ADD $8, R7
	VST1	[V19.D1], (R7); ADD $8, R7
	VST1	[V20.D1], (R7); ADD $8, R7
	VST1	[V21.D1], (R7); ADD $8, R7
	VST1	[V22.D1], (R7); ADD $8, R7
	VST1	[V23.D1], (R7); ADD $8, R7
	VST1	[V24.D1], (R7)

	// Closing permutation for the leaf lane.
	MOVD	$kt128_round_consts(SB), R1
	ADD	$96, R1
	KECCAK_12_ROUNDS

	// Extract the leaf CV (d[1] of lanes 0-3).
	VDUP	V0.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V1.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V2.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V3.D[1], V25.D2; VST1 [V25.D1], (R6)

	RET

// ABSORB_STRIPE_X1 XORs one 168-byte stripe from a single input pointer into
// the d[0] halves of state registers V0-V20. VLD1 of a .D1 vector zeroes the
// upper half of V25, so the d[1] halves absorb nothing.
#define ABSORB_STRIPE_X1(IN) \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V0.B16, V0.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V1.B16, V1.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V2.B16, V2.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V3.B16, V3.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V4.B16, V4.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V5.B16, V5.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V6.B16, V6.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V7.B16, V7.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V8.B16, V8.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V9.B16, V9.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V10.B16, V10.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V11.B16, V11.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V12.B16, V12.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V13.B16, V13.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V14.B16, V14.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V15.B16, V15.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V16.B16, V16.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V17.B16, V17.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V18.B16, V18.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V19.B16, V19.B16; \
	VLD1	(IN), [V25.D1]; ADD $8, IN; VEOR V25.B16, V20.B16, V20.B16

// func processLeafPairPartialARM64(in0, in1 *byte, nShared uint64, cv *byte, lane1 *uint64)
//
// Computes the complete leaf at in0's 32-byte CV while carrying a partial
// leaf's whole rate-blocks through the same permutations: for the first
// nShared stripes both lanes absorb 2-wide (in0 in d[0], in1 in d[1]); the
// partial lane's 25-lane state is then written to lane1 for the Go caller to
// finish (ragged tail, padding, and closing permutation), and the complete
// leaf runs its remaining 48-nShared stripes and padded final block 1-wide.
// Reads exactly 8192 bytes from in0 and nShared*168 from in1; nShared must be
// in [0, 48].
TEXT ·processLeafPairPartialARM64(SB), NOSPLIT, $0-40
	MOVD	in0+0(FP), R2
	MOVD	in1+8(FP), R3
	MOVD	nShared+16(FP), R4

	// Zero state V0-V24.
	VEOR	V0.B16, V0.B16, V0.B16
	VEOR	V1.B16, V1.B16, V1.B16
	VEOR	V2.B16, V2.B16, V2.B16
	VEOR	V3.B16, V3.B16, V3.B16
	VEOR	V4.B16, V4.B16, V4.B16
	VEOR	V5.B16, V5.B16, V5.B16
	VEOR	V6.B16, V6.B16, V6.B16
	VEOR	V7.B16, V7.B16, V7.B16
	VEOR	V8.B16, V8.B16, V8.B16
	VEOR	V9.B16, V9.B16, V9.B16
	VEOR	V10.B16, V10.B16, V10.B16
	VEOR	V11.B16, V11.B16, V11.B16
	VEOR	V12.B16, V12.B16, V12.B16
	VEOR	V13.B16, V13.B16, V13.B16
	VEOR	V14.B16, V14.B16, V14.B16
	VEOR	V15.B16, V15.B16, V15.B16
	VEOR	V16.B16, V16.B16, V16.B16
	VEOR	V17.B16, V17.B16, V17.B16
	VEOR	V18.B16, V18.B16, V18.B16
	VEOR	V19.B16, V19.B16, V19.B16
	VEOR	V20.B16, V20.B16, V20.B16
	VEOR	V21.B16, V21.B16, V21.B16
	VEOR	V22.B16, V22.B16, V22.B16
	VEOR	V23.B16, V23.B16, V23.B16
	VEOR	V24.B16, V24.B16, V24.B16

	MOVD	$48, R5
	SUB	R4, R5, R5		// 1-wide stripes after the shared pass

	CBZ	R4, partial_pair_export

partial_pair_shared_loop:
	ABSORB_STRIPE_X2(R2, R3)

	MOVD	$kt128_round_consts(SB), R1
	ADD	$96, R1
	KECCAK_12_ROUNDS

	SUBS	$1, R4
	BNE	partial_pair_shared_loop

partial_pair_export:
	// Export the partial lane's state (d[1] of all 25 lanes) for the Go tail
	// finish; the remaining 1-wide work only affects d[0].
	MOVD	lane1+32(FP), R6
	VDUP	V0.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V1.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V2.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V3.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V4.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V5.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V6.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V7.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V8.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V9.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V10.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V11.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V12.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V13.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V14.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V15.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V16.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V17.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V18.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V19.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V20.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V21.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V22.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V23.D[1], V25.D2; VST1 [V25.D1], (R6); ADD $8, R6
	VDUP	V24.D[1], V25.D2; VST1 [V25.D1], (R6)

	CBZ	R5, partial_pair_final

partial_pair_x1_loop:
	ABSORB_STRIPE_X1(R2)

	MOVD	$kt128_round_consts(SB), R1
	ADD	$96, R1
	KECCAK_12_ROUNDS

	SUBS	$1, R5
	BNE	partial_pair_x1_loop

partial_pair_final:
	// Absorb the complete leaf's final 16 lanes.
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V0.B16, V0.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V1.B16, V1.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V2.B16, V2.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V3.B16, V3.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V4.B16, V4.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V5.B16, V5.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V6.B16, V6.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V7.B16, V7.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V8.B16, V8.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V9.B16, V9.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V10.B16, V10.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V11.B16, V11.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V12.B16, V12.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V13.B16, V13.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V14.B16, V14.B16
	VLD1	(R2), [V25.D1]; ADD $8, R2; VEOR V25.B16, V15.B16, V15.B16

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20 (the
	// d[1] halves are dead after the export above).
	MOVD	$0x0B, R9
	VDUP	R9, V25.D2
	VEOR	V25.B16, V16.B16, V16.B16
	MOVD	$0x8000000000000000, R9
	VDUP	R9, V25.D2
	VEOR	V25.B16, V20.B16, V20.B16

	MOVD	$kt128_round_consts(SB), R1
	ADD	$96, R1
	KECCAK_12_ROUNDS

	// Extract the complete leaf's CV (d[0] of lanes 0-3).
	MOVD	cv+24(FP), R6
	VST1	[V0.D1], (R6); ADD $8, R6
	VST1	[V1.D1], (R6); ADD $8, R6
	VST1	[V2.D1], (R6); ADD $8, R6
	VST1	[V3.D1], (R6)

	RET

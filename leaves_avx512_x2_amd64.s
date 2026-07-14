// 2-wide KT128 leaf pair kernel — AVX-512VL (XMM) implementation.
//
// Processes 2 contiguous 8192-byte chunks with the two lanes packed into the
// qwords of X0-X24, absorbing via plain loads instead of the 8-wide kernels'
// gathers. A narrow pass costs far less than a masked 8-wide pass, so this
// gives small chunk counts arm64-pair economics on AVX-512 hosts.

//go:build !purego

#include "textflag.h"
#include "keccak_round_avx512_x2_amd64.h"

// ABSORB_LANE_X2 loads one uint64 from each chunk pointer (SI, DX) at the
// given byte offset, packs them {chunk0, chunk1}, and XORs into Xlane.
#define ABSORB_LANE_X2(offset, Xlane) \
	VMOVQ	offset(SI), X25; \
	VPINSRQ	$1, offset(DX), X25, X25; \
	VPXORQ	X25, Xlane, Xlane

// ZERO_STATE_X2 zeroes the Keccak state X0-X24.
#define ZERO_STATE_X2 \
	VPXORQ	X0, X0, X0; \
	VPXORQ	X1, X1, X1; \
	VPXORQ	X2, X2, X2; \
	VPXORQ	X3, X3, X3; \
	VPXORQ	X4, X4, X4; \
	VPXORQ	X5, X5, X5; \
	VPXORQ	X6, X6, X6; \
	VPXORQ	X7, X7, X7; \
	VPXORQ	X8, X8, X8; \
	VPXORQ	X9, X9, X9; \
	VPXORQ	X10, X10, X10; \
	VPXORQ	X11, X11, X11; \
	VPXORQ	X12, X12, X12; \
	VPXORQ	X13, X13, X13; \
	VPXORQ	X14, X14, X14; \
	VPXORQ	X15, X15, X15; \
	VPXORQ	X16, X16, X16; \
	VPXORQ	X17, X17, X17; \
	VPXORQ	X18, X18, X18; \
	VPXORQ	X19, X19, X19; \
	VPXORQ	X20, X20, X20; \
	VPXORQ	X21, X21, X21; \
	VPXORQ	X22, X22, X22; \
	VPXORQ	X23, X23, X23; \
	VPXORQ	X24, X24, X24

// ABSORB_STRIPE21_X2 absorbs one full 168-byte stripe (21 lanes) from both
// chunk pointers, advancing them.
#define ABSORB_STRIPE21_X2 \
	ABSORB_LANE_X2(0*8, X0); \
	ABSORB_LANE_X2(1*8, X1); \
	ABSORB_LANE_X2(2*8, X2); \
	ABSORB_LANE_X2(3*8, X3); \
	ABSORB_LANE_X2(4*8, X4); \
	ABSORB_LANE_X2(5*8, X5); \
	ABSORB_LANE_X2(6*8, X6); \
	ABSORB_LANE_X2(7*8, X7); \
	ABSORB_LANE_X2(8*8, X8); \
	ABSORB_LANE_X2(9*8, X9); \
	ABSORB_LANE_X2(10*8, X10); \
	ABSORB_LANE_X2(11*8, X11); \
	ABSORB_LANE_X2(12*8, X12); \
	ABSORB_LANE_X2(13*8, X13); \
	ABSORB_LANE_X2(14*8, X14); \
	ABSORB_LANE_X2(15*8, X15); \
	ABSORB_LANE_X2(16*8, X16); \
	ABSORB_LANE_X2(17*8, X17); \
	ABSORB_LANE_X2(18*8, X18); \
	ABSORB_LANE_X2(19*8, X19); \
	ABSORB_LANE_X2(20*8, X20); \
	ADDQ	$168, SI; \
	ADDQ	$168, DX

// ABSORB_FINAL16_X2 absorbs the final 128-byte partial block (16 lanes) from
// both chunk pointers.
#define ABSORB_FINAL16_X2 \
	ABSORB_LANE_X2(0*8, X0); \
	ABSORB_LANE_X2(1*8, X1); \
	ABSORB_LANE_X2(2*8, X2); \
	ABSORB_LANE_X2(3*8, X3); \
	ABSORB_LANE_X2(4*8, X4); \
	ABSORB_LANE_X2(5*8, X5); \
	ABSORB_LANE_X2(6*8, X6); \
	ABSORB_LANE_X2(7*8, X7); \
	ABSORB_LANE_X2(8*8, X8); \
	ABSORB_LANE_X2(9*8, X9); \
	ABSORB_LANE_X2(10*8, X10); \
	ABSORB_LANE_X2(11*8, X11); \
	ABSORB_LANE_X2(12*8, X12); \
	ABSORB_LANE_X2(13*8, X13); \
	ABSORB_LANE_X2(14*8, X14); \
	ABSORB_LANE_X2(15*8, X15)

// PERMUTE12_X2 runs the 12-round Keccak-p[1600,12] permutation on X0-X24.
#define PERMUTE12_X2 \
	LEAQ	kt128_round_consts_2x+192(SB), R11; \
	X2_4ROUNDS(0, 16, 32, 48); \
	X2_4ROUNDS(64, 80, 96, 112); \
	X2_4ROUNDS(128, 144, 160, 176)

// func processLeavesPairAVX512(input *byte, cvs *byte)
//
// Processes 2 contiguous 8192-byte chunks, writing 2 × 32-byte CVs to cvs.
TEXT ·processLeavesPairAVX512(SB), NOSPLIT, $0-16
	MOVQ	input+0(FP), SI
	LEAQ	8192(SI), DX
	MOVQ	cvs+8(FP), DI

	// Zero state X0-X24.
	ZERO_STATE_X2

	MOVQ	$48, R12

leaves_pair_avx512_loop:
	ABSORB_STRIPE21_X2

	PERMUTE12_X2

	SUBQ	$1, R12
	JNZ	leaves_pair_avx512_loop

	// Absorb final 16 lanes (128-byte remainder).
	ABSORB_FINAL16_X2

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20.
	MOVQ	$0x0B, AX
	VPBROADCASTQ	AX, X25
	VPXORQ	X25, X16, X16
	MOVQ	$0x8000000000000000, AX
	VPBROADCASTQ	AX, X25
	VPXORQ	X25, X20, X20

	// Final permutation.
	PERMUTE12_X2

	// Extract CVs: chunk 0 from the low qwords of lanes 0-3, chunk 1 from
	// the high qwords.
	VMOVQ	X0, 0(DI)
	VMOVQ	X1, 8(DI)
	VMOVQ	X2, 16(DI)
	VMOVQ	X3, 24(DI)
	VPEXTRQ	$1, X0, 32(DI)
	VPEXTRQ	$1, X1, 40(DI)
	VPEXTRQ	$1, X2, 48(DI)
	VPEXTRQ	$1, X3, 56(DI)

	RET

// func processS0LeafPairAVX512(input *byte, state *uint64, cv *byte)
//
// Fused S_0 + first-leaf pair, the 2-wide analog of processS0LeavesAVX512's
// n=2 case at pair cost instead of a flat masked pass. S_0 || kt12 marker has
// the same permutation schedule as a leaf (48 full stripes plus a 128-byte
// remainder and one more lane), so both ride one XMM pass: the final node in
// the low qwords, the leaf in the high. They differ only in the last block:
// the final node XORs the marker word 0x03 at lane 16 and is NOT permuted
// (its 25-lane state is written to state first; position 136 is set by the
// Go wrapper), while the leaf XORs DS 0x0B at lane 16 and pad10*1 0x80 at
// lane 20 and is; its 32-byte CV is written to cv.
TEXT ·processS0LeafPairAVX512(SB), NOSPLIT, $0-24
	MOVQ	input+0(FP), SI
	LEAQ	8192(SI), DX
	MOVQ	state+8(FP), R8
	MOVQ	cv+16(FP), DI

	// Zero state X0-X24.
	ZERO_STATE_X2

	MOVQ	$48, R12

s0leaf_pair_avx512_loop:
	ABSORB_STRIPE21_X2

	PERMUTE12_X2

	SUBQ	$1, R12
	JNZ	s0leaf_pair_avx512_loop

	// Absorb final 16 lanes (128-byte remainder).
	ABSORB_FINAL16_X2

	// Lane 16: kt12 marker word 0x03 for the final node (low), leaf DS 0x0B
	// for the leaf (high).
	MOVQ	$0x03, AX
	VMOVQ	AX, X25
	MOVQ	$0x0B, AX
	VPINSRQ	$1, AX, X25, X25
	VPXORQ	X25, X16, X16

	// Lane 20: pad10*1 end 0x80 for the leaf only.
	VPXORQ	X25, X25, X25
	MOVQ	$0x8000000000000000, AX
	VPINSRQ	$1, AX, X25, X25
	VPXORQ	X25, X20, X20

	// Extract the final-node state (low qwords of all 25 lanes) before the
	// leaf's closing permutation scrambles it.
	VMOVQ	X0, 0(R8)
	VMOVQ	X1, 8(R8)
	VMOVQ	X2, 16(R8)
	VMOVQ	X3, 24(R8)
	VMOVQ	X4, 32(R8)
	VMOVQ	X5, 40(R8)
	VMOVQ	X6, 48(R8)
	VMOVQ	X7, 56(R8)
	VMOVQ	X8, 64(R8)
	VMOVQ	X9, 72(R8)
	VMOVQ	X10, 80(R8)
	VMOVQ	X11, 88(R8)
	VMOVQ	X12, 96(R8)
	VMOVQ	X13, 104(R8)
	VMOVQ	X14, 112(R8)
	VMOVQ	X15, 120(R8)
	VMOVQ	X16, 128(R8)
	VMOVQ	X17, 136(R8)
	VMOVQ	X18, 144(R8)
	VMOVQ	X19, 152(R8)
	VMOVQ	X20, 160(R8)
	VMOVQ	X21, 168(R8)
	VMOVQ	X22, 176(R8)
	VMOVQ	X23, 184(R8)
	VMOVQ	X24, 192(R8)

	// Closing permutation for the leaf lane.
	PERMUTE12_X2

	// Extract the leaf CV (high qwords of lanes 0-3).
	VPEXTRQ	$1, X0, 0(DI)
	VPEXTRQ	$1, X1, 8(DI)
	VPEXTRQ	$1, X2, 16(DI)
	VPEXTRQ	$1, X3, 24(DI)

	RET

// ABSORB_LANE_X1LOW XORs one uint64 from the chunk pointer SI into the low
// qword of Xlane; VMOVQ zeroes the scratch high qword, so the (dead) high
// lane absorbs nothing.
#define ABSORB_LANE_X1LOW(offset, Xlane) \
	VMOVQ	offset(SI), X25; \
	VPXORQ	X25, Xlane, Xlane

// func processLeafPairPartialAVX512(in0, in1 *byte, nShared uint64, cv *byte, lane1 *uint64)
//
// Computes the complete leaf at in0's 32-byte CV while carrying a partial
// leaf at in1 through its nShared whole rate-blocks in the same XMM
// permutations: complete leaf in the low qwords, partial in the high. After
// the shared stripes the partial lane's 25-lane state is written to lane1
// for the Go caller to finish (ragged tail, padding, closing permutation),
// and the complete leaf runs its remaining 48-nShared stripes and padded
// final block on the low qwords alone. Reads exactly 8192 bytes from in0 and
// nShared*168 from in1; nShared must be in [0, 48].
TEXT ·processLeafPairPartialAVX512(SB), NOSPLIT, $0-40
	MOVQ	in0+0(FP), SI
	MOVQ	in1+8(FP), DX
	MOVQ	nShared+16(FP), R13
	MOVQ	cv+24(FP), DI
	MOVQ	lane1+32(FP), R8

	// Zero state X0-X24.
	ZERO_STATE_X2

	MOVQ	$48, R12
	SUBQ	R13, R12	// 1-wide stripes after the shared pass

	TESTQ	R13, R13
	JZ	pair_partial_export

pair_partial_shared_loop:
	ABSORB_STRIPE21_X2

	PERMUTE12_X2

	SUBQ	$1, R13
	JNZ	pair_partial_shared_loop

pair_partial_export:
	// Export the partial lane's state (high qwords of all 25 lanes) for the
	// Go tail finish; the remaining 1-wide work only touches the low qwords.
	VPEXTRQ	$1, X0, 0(R8)
	VPEXTRQ	$1, X1, 8(R8)
	VPEXTRQ	$1, X2, 16(R8)
	VPEXTRQ	$1, X3, 24(R8)
	VPEXTRQ	$1, X4, 32(R8)
	VPEXTRQ	$1, X5, 40(R8)
	VPEXTRQ	$1, X6, 48(R8)
	VPEXTRQ	$1, X7, 56(R8)
	VPEXTRQ	$1, X8, 64(R8)
	VPEXTRQ	$1, X9, 72(R8)
	VPEXTRQ	$1, X10, 80(R8)
	VPEXTRQ	$1, X11, 88(R8)
	VPEXTRQ	$1, X12, 96(R8)
	VPEXTRQ	$1, X13, 104(R8)
	VPEXTRQ	$1, X14, 112(R8)
	VPEXTRQ	$1, X15, 120(R8)
	VPEXTRQ	$1, X16, 128(R8)
	VPEXTRQ	$1, X17, 136(R8)
	VPEXTRQ	$1, X18, 144(R8)
	VPEXTRQ	$1, X19, 152(R8)
	VPEXTRQ	$1, X20, 160(R8)
	VPEXTRQ	$1, X21, 168(R8)
	VPEXTRQ	$1, X22, 176(R8)
	VPEXTRQ	$1, X23, 184(R8)
	VPEXTRQ	$1, X24, 192(R8)

	TESTQ	R12, R12
	JZ	pair_partial_final

pair_partial_rest_loop:
	ABSORB_LANE_X1LOW(0*8, X0)
	ABSORB_LANE_X1LOW(1*8, X1)
	ABSORB_LANE_X1LOW(2*8, X2)
	ABSORB_LANE_X1LOW(3*8, X3)
	ABSORB_LANE_X1LOW(4*8, X4)
	ABSORB_LANE_X1LOW(5*8, X5)
	ABSORB_LANE_X1LOW(6*8, X6)
	ABSORB_LANE_X1LOW(7*8, X7)
	ABSORB_LANE_X1LOW(8*8, X8)
	ABSORB_LANE_X1LOW(9*8, X9)
	ABSORB_LANE_X1LOW(10*8, X10)
	ABSORB_LANE_X1LOW(11*8, X11)
	ABSORB_LANE_X1LOW(12*8, X12)
	ABSORB_LANE_X1LOW(13*8, X13)
	ABSORB_LANE_X1LOW(14*8, X14)
	ABSORB_LANE_X1LOW(15*8, X15)
	ABSORB_LANE_X1LOW(16*8, X16)
	ABSORB_LANE_X1LOW(17*8, X17)
	ABSORB_LANE_X1LOW(18*8, X18)
	ABSORB_LANE_X1LOW(19*8, X19)
	ABSORB_LANE_X1LOW(20*8, X20)
	ADDQ	$168, SI

	PERMUTE12_X2

	SUBQ	$1, R12
	JNZ	pair_partial_rest_loop

pair_partial_final:
	// Absorb the complete leaf's final 16 lanes.
	ABSORB_LANE_X1LOW(0*8, X0)
	ABSORB_LANE_X1LOW(1*8, X1)
	ABSORB_LANE_X1LOW(2*8, X2)
	ABSORB_LANE_X1LOW(3*8, X3)
	ABSORB_LANE_X1LOW(4*8, X4)
	ABSORB_LANE_X1LOW(5*8, X5)
	ABSORB_LANE_X1LOW(6*8, X6)
	ABSORB_LANE_X1LOW(7*8, X7)
	ABSORB_LANE_X1LOW(8*8, X8)
	ABSORB_LANE_X1LOW(9*8, X9)
	ABSORB_LANE_X1LOW(10*8, X10)
	ABSORB_LANE_X1LOW(11*8, X11)
	ABSORB_LANE_X1LOW(12*8, X12)
	ABSORB_LANE_X1LOW(13*8, X13)
	ABSORB_LANE_X1LOW(14*8, X14)
	ABSORB_LANE_X1LOW(15*8, X15)

	// XOR padding: DS=0x0B into lane 16, pad10*1 end 0x80 into lane 20 (the
	// high qwords are dead after the export above).
	MOVQ	$0x0B, AX
	VPBROADCASTQ	AX, X25
	VPXORQ	X25, X16, X16
	MOVQ	$0x8000000000000000, AX
	VPBROADCASTQ	AX, X25
	VPXORQ	X25, X20, X20

	// Final permutation.
	PERMUTE12_X2

	// Extract the complete leaf's CV (low qwords of lanes 0-3).
	VMOVQ	X0, 0(DI)
	VMOVQ	X1, 8(DI)
	VMOVQ	X2, 16(DI)
	VMOVQ	X3, 24(DI)

	RET

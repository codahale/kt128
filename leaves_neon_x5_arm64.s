// Hybrid scalar/NEON 5- and 3-leaf kernels — ARM64.
//
// Processes 5 × 8192-byte chunks per call: chunks 0-3 as two sequential
// 2-wide NEON pair passes and chunk 4 on the scalar pipes, woven into the
// NEON round stream at a 1:2 rate (six scalar rounds inside each 12-round
// NEON permute, so one scalar block completes per two NEON block
// iterations). A leaf is 48 full 168-byte stripes plus one padded 128-byte
// final block — 49 permutes — so the two passes are 2 × 49 = 98 NEON
// iterations, during which the scalar lane completes exactly 49 blocks: one
// chunk. The scalar stream executes almost entirely in the shadow of the
// NEON stream, so the fifth chunk is nearly free.
//
// NEON iterations alternate two phases: A hosts the scalar block absorb plus
// scalar rounds 0-5, B hosts scalar rounds 6-11. 49 is odd, so the phase
// alternation crosses the pass boundary: the pass-1 epilogue (CV extraction
// and re-zeroing the pair state) runs between a scalar block's A and B
// phases, touching only R22/R23 and the vector registers.
//
// Scalar state register map and SROUND/SCALAR_ABSORB_* macros: see
// keccak_round_scalar_arm64.h. The NEON pair I/O loads its two walking pointers
// from the frame each iteration (SROUND owns R22-R26 between blocks), and
// unlike TurboSHAKE's 167-byte-rate cousins there is no partial lane: 168 is
// 21 whole lanes and the final block is 16 whole lanes, so every load is an
// aligned-stride 8-byte read and the walkers end exactly at their chunks'
// ends.
//
// Frame: 0=src0, 8=src1 (NEON walkers), 32/40/48/56/64=spilled scalar lanes
// (fixed by keccak_round_scalar_arm64.h), 72=scalar src walker, 88=unit counter,
// 96=cvs.

//go:build !purego

#include "textflag.h"
#include "keccak_round_neon_x2_arm64.h"
#include "keccak_round_scalar_arm64.h"

// ABSORB_LANE_X5 XORs one 8-byte lane from each of the two NEON walking
// pointers (R22, R23) into state register VK, packed {in0, in1}.
#define ABSORB_LANE_X5(VK) \
	VLD1	(R22), [V25.D1]; ADD $8, R22; VLD1 (R23), [V26.D1]; ADD $8, R23; VZIP1 V26.D2, V25.D2, V25.D2; VEOR V25.B16, VK.B16, VK.B16

// RC_RESET points R1 at the 12-round tail of the round-constant table for
// the next NEON permute.
#define RC_RESET \
	MOVD	$kt128_round_consts(SB), R1; \
	ADD	$96, R1

// NEON_IO_MORE absorbs one full 168-byte stripe (21 lanes) into both NEON
// instances, walking the frame-resident pointers.
#define NEON_IO_MORE \
	LDP	0(RSP), (R22, R23); \
	ABSORB_LANE_X5(V0); \
	ABSORB_LANE_X5(V1); \
	ABSORB_LANE_X5(V2); \
	ABSORB_LANE_X5(V3); \
	ABSORB_LANE_X5(V4); \
	ABSORB_LANE_X5(V5); \
	ABSORB_LANE_X5(V6); \
	ABSORB_LANE_X5(V7); \
	ABSORB_LANE_X5(V8); \
	ABSORB_LANE_X5(V9); \
	ABSORB_LANE_X5(V10); \
	ABSORB_LANE_X5(V11); \
	ABSORB_LANE_X5(V12); \
	ABSORB_LANE_X5(V13); \
	ABSORB_LANE_X5(V14); \
	ABSORB_LANE_X5(V15); \
	ABSORB_LANE_X5(V16); \
	ABSORB_LANE_X5(V17); \
	ABSORB_LANE_X5(V18); \
	ABSORB_LANE_X5(V19); \
	ABSORB_LANE_X5(V20); \
	STP	(R22, R23), 0(RSP); \
	RC_RESET

// NEON_IO_LAST absorbs the final 128-byte partial block (16 lanes) and XORs
// the leaf padding: DS 0x0B into lane 16, pad10*1 end 0x80 into lane 20.
#define NEON_IO_LAST \
	LDP	0(RSP), (R22, R23); \
	ABSORB_LANE_X5(V0); \
	ABSORB_LANE_X5(V1); \
	ABSORB_LANE_X5(V2); \
	ABSORB_LANE_X5(V3); \
	ABSORB_LANE_X5(V4); \
	ABSORB_LANE_X5(V5); \
	ABSORB_LANE_X5(V6); \
	ABSORB_LANE_X5(V7); \
	ABSORB_LANE_X5(V8); \
	ABSORB_LANE_X5(V9); \
	ABSORB_LANE_X5(V10); \
	ABSORB_LANE_X5(V11); \
	ABSORB_LANE_X5(V12); \
	ABSORB_LANE_X5(V13); \
	ABSORB_LANE_X5(V14); \
	ABSORB_LANE_X5(V15); \
	STP	(R22, R23), 0(RSP); \
	MOVD	$0x0B, R22; \
	VDUP	R22, V25.D2; \
	VEOR	V25.B16, V16.B16, V16.B16; \
	MOVD	$0x8000000000000000, R22; \
	VDUP	R22, V25.D2; \
	VEOR	V25.B16, V20.B16, V20.B16; \
	RC_RESET

// Scalar block absorbs at the hybrid frame offset (72=src walker).
#define SCALAR_IO_MORE \
	MOVD	72(RSP), R25; \
	SCALAR_ABSORB_LANES21; \
	MOVD	R25, 72(RSP)

#define SCALAR_IO_LAST \
	MOVD	72(RSP), R25; \
	SCALAR_ABSORB_TAIL16_LAST; \
	MOVD	R25, 72(RSP)

// WEAVE_A runs the 12-round NEON permute with scalar rounds 0-5 woven in
// after every second NEON round; WEAVE_B hosts scalar rounds 6-11. Two
// consecutive weaves complete one scalar permute. The round constants are
// RC[12..23] of Keccak-p[1600].
#define WEAVE_A \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x000000008000808B); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x800000000000008B); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x8000000000008089); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x8000000000008003); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x8000000000008002); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x8000000000000080)

#define WEAVE_B \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x000000000000800A); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x800000008000000A); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x8000000080008081); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x8000000000008080); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x0000000080000001); \
	KECCAK_ROUND; \
	KECCAK_ROUND; \
	SROUND($0x8000000080008008)

// Scalar-only completion used by the x3 hybrid after its single NEON pair
// pass ends halfway through a scalar permutation.
#define SCALAR_ROUNDS_B_ONLY \
	SROUND($0x000000000000800A); \
	SROUND($0x800000008000000A); \
	SROUND($0x8000000080008081); \
	SROUND($0x8000000000008080); \
	SROUND($0x0000000080000001); \
	SROUND($0x8000000080008008)

// EXTRACT_CVS_X5 writes lanes 0-3 of both NEON instances to the CV buffer at
// R22 (64 bytes): instance 0 from the low halves, instance 1 from the high.
#define EXTRACT_CVS_X5 \
	VST1	[V0.D1], (R22); ADD $8, R22; \
	VST1	[V1.D1], (R22); ADD $8, R22; \
	VST1	[V2.D1], (R22); ADD $8, R22; \
	VST1	[V3.D1], (R22); ADD $8, R22; \
	VDUP	V0.D[1], V25.D2; VST1 [V25.D1], (R22); ADD $8, R22; \
	VDUP	V1.D[1], V25.D2; VST1 [V25.D1], (R22); ADD $8, R22; \
	VDUP	V2.D[1], V25.D2; VST1 [V25.D1], (R22); ADD $8, R22; \
	VDUP	V3.D[1], V25.D2; VST1 [V25.D1], (R22)

// ZERO_NEON_STATE zeroes the NEON pair state V0-V24.
#define ZERO_NEON_STATE \
	VEOR	V0.B16, V0.B16, V0.B16; \
	VEOR	V1.B16, V1.B16, V1.B16; \
	VEOR	V2.B16, V2.B16, V2.B16; \
	VEOR	V3.B16, V3.B16, V3.B16; \
	VEOR	V4.B16, V4.B16, V4.B16; \
	VEOR	V5.B16, V5.B16, V5.B16; \
	VEOR	V6.B16, V6.B16, V6.B16; \
	VEOR	V7.B16, V7.B16, V7.B16; \
	VEOR	V8.B16, V8.B16, V8.B16; \
	VEOR	V9.B16, V9.B16, V9.B16; \
	VEOR	V10.B16, V10.B16, V10.B16; \
	VEOR	V11.B16, V11.B16, V11.B16; \
	VEOR	V12.B16, V12.B16, V12.B16; \
	VEOR	V13.B16, V13.B16, V13.B16; \
	VEOR	V14.B16, V14.B16, V14.B16; \
	VEOR	V15.B16, V15.B16, V15.B16; \
	VEOR	V16.B16, V16.B16, V16.B16; \
	VEOR	V17.B16, V17.B16, V17.B16; \
	VEOR	V18.B16, V18.B16, V18.B16; \
	VEOR	V19.B16, V19.B16, V19.B16; \
	VEOR	V20.B16, V20.B16, V20.B16; \
	VEOR	V21.B16, V21.B16, V21.B16; \
	VEOR	V22.B16, V22.B16, V22.B16; \
	VEOR	V23.B16, V23.B16, V23.B16; \
	VEOR	V24.B16, V24.B16, V24.B16

// X5_PASS1_EPILOGUE extracts pair-1 CVs (cvs[0:64]), re-zeroes the pair
// state, and derives the pair-2 pointers from the walked src1 pointer (chunk
// 1's end is chunk 2's start). Runs between the A and B phases of scalar
// block 25; touches only R22/R23 and the vector registers.
#define X5_PASS1_EPILOGUE \
	MOVD	96(RSP), R22; \
	EXTRACT_CVS_X5; \
	ZERO_NEON_STATE; \
	MOVD	8(RSP), R22; \
	ADD	$8192, R22, R23; \
	STP	(R22, R23), 0(RSP)

// X5_EPILOGUE extracts pair-2 CVs (cvs[64:128]) and the scalar leaf CV
// (cvs[128:160], scalar lanes 0-3).
#define X5_EPILOGUE \
	MOVD	96(RSP), R22; \
	ADD	$64, R22; \
	EXTRACT_CVS_X5; \
	MOVD	96(RSP), R22; \
	MOVD	R0, 128(R22); \
	MOVD	64(RSP), R23; \
	MOVD	R23, 136(R22); \
	MOVD	R2, 144(R22); \
	MOVD	R3, 152(R22)

#define X5_COUNT(n) \
	MOVD	$n, R22; \
	MOVD	R22, 88(RSP)

#define X5_DEC_COUNT(label) \
	MOVD	88(RSP), R22; \
	SUB	$1, R22; \
	MOVD	R22, 88(RSP); \
	CBNZ	R22, label

// func processLeaves5ARM64(input *byte, cvs *byte)
TEXT ·processLeaves5ARM64(SB), NOSPLIT, $104-16
	MOVD	input+0(FP), R22
	MOVD	cvs+8(FP), R23
	MOVD	R23, 96(RSP)
	ADD	$32768, R22, R23	// scalar walker: chunk 4
	MOVD	R23, 72(RSP)
	ADD	$8192, R22, R23		// NEON walkers: chunks 0, 1
	STP	(R22, R23), 0(RSP)

	// Zero the scalar state: register lanes and spilled lanes.
	MOVD	ZR, R0
	MOVD	ZR, R2
	MOVD	ZR, R3
	MOVD	ZR, R4
	MOVD	ZR, R5
	MOVD	ZR, R6
	MOVD	ZR, R7
	MOVD	ZR, R8
	MOVD	ZR, R9
	MOVD	ZR, R10
	MOVD	ZR, R11
	MOVD	ZR, R12
	MOVD	ZR, R13
	MOVD	ZR, R14
	MOVD	ZR, R15
	MOVD	ZR, R16
	MOVD	ZR, R17
	MOVD	ZR, R19
	MOVD	ZR, R20
	MOVD	ZR, R21
	MOVD	ZR, 32(RSP)
	MOVD	ZR, 40(RSP)
	MOVD	ZR, 48(RSP)
	MOVD	ZR, 56(RSP)
	MOVD	ZR, 64(RSP)

	ZERO_NEON_STATE

	// Pass 1, iterations 1-48: 24 (A,B) units over chunks 0,1; scalar
	// blocks 1-24 complete inside their units.
	X5_COUNT(24)
x5_pass1_loop:
	NEON_IO_MORE
	SCALAR_IO_MORE
	WEAVE_A
	NEON_IO_MORE
	WEAVE_B
	X5_DEC_COUNT(x5_pass1_loop)

	// Iteration 49: NEON final block for chunks 0,1; scalar block 25 A-phase.
	NEON_IO_LAST
	SCALAR_IO_MORE
	WEAVE_A

	X5_PASS1_EPILOGUE

	// Iteration 50: first block of chunks 2,3; scalar block 25 B-phase.
	NEON_IO_MORE
	WEAVE_B

	// Pass 2, iterations 51-96: 23 units; scalar blocks 26-48.
	X5_COUNT(23)
x5_pass2_loop:
	NEON_IO_MORE
	SCALAR_IO_MORE
	WEAVE_A
	NEON_IO_MORE
	WEAVE_B
	X5_DEC_COUNT(x5_pass2_loop)

	// Iteration 97: scalar final block (49, padded) A-phase.
	NEON_IO_MORE
	SCALAR_IO_LAST
	WEAVE_A

	// Iteration 98: NEON final block for chunks 2,3; scalar block 49 B-phase.
	NEON_IO_LAST
	WEAVE_B

	X5_EPILOGUE
	RET

// func processLeaves3ARM64(pairInput, scalarInput *byte, cvs *byte, state *uint64)
//
// Processes pairInput as one x2 NEON pair while scalarInput advances on the
// scalar pipes. The pair's 49 iterations host 24.5 scalar permutations; the
// exported scalar state has absorbed 25 complete rate blocks.
TEXT ·processLeaves3ARM64(SB), NOSPLIT, $104-32
	MOVD	pairInput+0(FP), R22
	MOVD	scalarInput+8(FP), R23
	MOVD	R23, 72(RSP)
	MOVD	cvs+16(FP), R23
	MOVD	R23, 96(RSP)
	MOVD	state+24(FP), R23
	MOVD	R23, 80(RSP)
	ADD	$8192, R22, R23		// NEON pair walkers
	STP	(R22, R23), 0(RSP)

	// Zero the scalar state: register lanes and spilled lanes.
	MOVD	ZR, R0
	MOVD	ZR, R2
	MOVD	ZR, R3
	MOVD	ZR, R4
	MOVD	ZR, R5
	MOVD	ZR, R6
	MOVD	ZR, R7
	MOVD	ZR, R8
	MOVD	ZR, R9
	MOVD	ZR, R10
	MOVD	ZR, R11
	MOVD	ZR, R12
	MOVD	ZR, R13
	MOVD	ZR, R14
	MOVD	ZR, R15
	MOVD	ZR, R16
	MOVD	ZR, R17
	MOVD	ZR, R19
	MOVD	ZR, R20
	MOVD	ZR, R21
	MOVD	ZR, 32(RSP)
	MOVD	ZR, 40(RSP)
	MOVD	ZR, 48(RSP)
	MOVD	ZR, 56(RSP)
	MOVD	ZR, 64(RSP)

	ZERO_NEON_STATE

	// NEON iterations 1-48 complete pair blocks 1-48 and scalar blocks 1-24.
	X5_COUNT(24)
x3_pair_loop:
	NEON_IO_MORE
	SCALAR_IO_MORE
	WEAVE_A
	NEON_IO_MORE
	WEAVE_B
	X5_DEC_COUNT(x3_pair_loop)

	// NEON iteration 49 closes the pair and advances scalar block 25 halfway.
	NEON_IO_LAST
	SCALAR_IO_MORE
	WEAVE_A

	// Finish scalar block 25. The remaining blocks use the optimized x1 loop.
	SCALAR_ROUNDS_B_ONLY

	// Export the scalar state after 25 complete rate blocks.
	MOVD	80(RSP), R22
	MOVD	R0, 0(R22)
	MOVD	64(RSP), R23
	MOVD	R23, 8(R22)
	MOVD	R2, 16(R22)
	MOVD	R3, 24(R22)
	MOVD	R4, 32(R22)
	MOVD	R5, 40(R22)
	MOVD	32(RSP), R23
	MOVD	R23, 48(R22)
	MOVD	R6, 56(R22)
	MOVD	R7, 64(R22)
	MOVD	R8, 72(R22)
	MOVD	R9, 80(R22)
	MOVD	40(RSP), R23
	MOVD	R23, 88(R22)
	MOVD	R10, 96(R22)
	MOVD	R11, 104(R22)
	MOVD	R12, 112(R22)
	MOVD	R13, 120(R22)
	MOVD	48(RSP), R23
	MOVD	R23, 128(R22)
	MOVD	R14, 136(R22)
	MOVD	R15, 144(R22)
	MOVD	R16, 152(R22)
	MOVD	R17, 160(R22)
	MOVD	56(RSP), R23
	MOVD	R23, 168(R22)
	MOVD	R19, 176(R22)
	MOVD	R20, 184(R22)
	MOVD	R21, 192(R22)

	// Extract the completed pair CVs.
	MOVD	96(RSP), R22
	EXTRACT_CVS_X5
	RET

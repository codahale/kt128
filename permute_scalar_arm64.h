// Scalar Keccak-p[1600] round for the hybrid scalar/NEON leaf kernel.
// Register/stack contract: state lanes A0=R0, A2-A5=R2-R5, A7-A10=R6-R9,
// A12-A15=R10-R13, A17-A20=R14-R17, A22-A24=R19-R21; spilled lanes A1=64(RSP),
// A6=32(RSP), A11=40(RSP), A16=48(RSP), A21=56(RSP); temps R22-R26. R1 and all
// vector registers are untouched, so the macros can interleave with the NEON
// KECCAK_ROUND stream.

// SROUND performs one Keccak-p[1600] round on the scalar state with round
// constant rc.
//
// Theta computes the five column parities into R22-R26, then applies
// D[x] = C[x-1] ^ rol(C[x+1], 1) lane by lane as two EORs (the second with a
// rotated operand), so no D registers are needed. The register-resident
// classes go first; the spilled class (x=1) goes last, when C1/C3/C4 are dead
// and R23 is free as scratch.
//
// Rho and pi run the standard 24-step move cycle with alternating temps
// R22/R23; rol(t, r) is ORR t@>(64-r), ZR. Spilled steps bounce through R24.
//
// Chi processes each row with b0 saved in R22 and the memory-resident b1
// loaded into R23; results for register lanes are computed in place, the b1
// result bounces through R24.
#define SROUND(rc) \
	/* theta: column parities */ \
	EOR	R5, R0, R22; \
	EOR	R9, R22, R22; \
	EOR	R13, R22, R22; \
	EOR	R17, R22, R22; \
	MOVD	64(RSP), R23; \
	MOVD	32(RSP), R24; \
	EOR	R24, R23, R23; \
	MOVD	40(RSP), R24; \
	EOR	R24, R23, R23; \
	MOVD	48(RSP), R24; \
	EOR	R24, R23, R23; \
	MOVD	56(RSP), R24; \
	EOR	R24, R23, R23; \
	EOR	R6, R2, R24; \
	EOR	R10, R24, R24; \
	EOR	R14, R24, R24; \
	EOR	R19, R24, R24; \
	EOR	R7, R3, R25; \
	EOR	R11, R25, R25; \
	EOR	R15, R25, R25; \
	EOR	R20, R25, R25; \
	EOR	R8, R4, R26; \
	EOR	R12, R26, R26; \
	EOR	R16, R26, R26; \
	EOR	R21, R26, R26; \
	/* theta: apply x=0 (C4, rol C1) */ \
	EOR	R26, R0, R0; \
	EOR	R23@>63, R0, R0; \
	EOR	R26, R5, R5; \
	EOR	R23@>63, R5, R5; \
	EOR	R26, R9, R9; \
	EOR	R23@>63, R9, R9; \
	EOR	R26, R13, R13; \
	EOR	R23@>63, R13, R13; \
	EOR	R26, R17, R17; \
	EOR	R23@>63, R17, R17; \
	/* theta: apply x=2 (C1, rol C3) */ \
	EOR	R23, R2, R2; \
	EOR	R25@>63, R2, R2; \
	EOR	R23, R6, R6; \
	EOR	R25@>63, R6, R6; \
	EOR	R23, R10, R10; \
	EOR	R25@>63, R10, R10; \
	EOR	R23, R14, R14; \
	EOR	R25@>63, R14, R14; \
	EOR	R23, R19, R19; \
	EOR	R25@>63, R19, R19; \
	/* theta: apply x=3 (C2, rol C4) */ \
	EOR	R24, R3, R3; \
	EOR	R26@>63, R3, R3; \
	EOR	R24, R7, R7; \
	EOR	R26@>63, R7, R7; \
	EOR	R24, R11, R11; \
	EOR	R26@>63, R11, R11; \
	EOR	R24, R15, R15; \
	EOR	R26@>63, R15, R15; \
	EOR	R24, R20, R20; \
	EOR	R26@>63, R20, R20; \
	/* theta: apply x=4 (C3, rol C0) */ \
	EOR	R25, R4, R4; \
	EOR	R22@>63, R4, R4; \
	EOR	R25, R8, R8; \
	EOR	R22@>63, R8, R8; \
	EOR	R25, R12, R12; \
	EOR	R22@>63, R12, R12; \
	EOR	R25, R16, R16; \
	EOR	R22@>63, R16, R16; \
	EOR	R25, R21, R21; \
	EOR	R22@>63, R21, R21; \
	/* theta: apply x=1 (C0, rol C2) to the spilled lanes */ \
	MOVD	64(RSP), R23; \
	EOR	R22, R23, R23; \
	EOR	R24@>63, R23, R23; \
	MOVD	R23, 64(RSP); \
	MOVD	32(RSP), R23; \
	EOR	R22, R23, R23; \
	EOR	R24@>63, R23, R23; \
	MOVD	R23, 32(RSP); \
	MOVD	40(RSP), R23; \
	EOR	R22, R23, R23; \
	EOR	R24@>63, R23, R23; \
	MOVD	R23, 40(RSP); \
	MOVD	48(RSP), R23; \
	EOR	R22, R23, R23; \
	EOR	R24@>63, R23, R23; \
	MOVD	R23, 48(RSP); \
	MOVD	56(RSP), R23; \
	EOR	R22, R23, R23; \
	EOR	R24@>63, R23, R23; \
	MOVD	R23, 56(RSP); \
	/* rho and pi: t = A1, then the 24-step cycle */ \
	MOVD	64(RSP), R22; \
	MOVD	R9, R23; \
	ORR	R22@>63, ZR, R9; \
	MOVD	R6, R22; \
	ORR	R23@>61, ZR, R6; \
	MOVD	40(RSP), R23; \
	ORR	R22@>58, ZR, R24; \
	MOVD	R24, 40(RSP); \
	MOVD	R14, R22; \
	ORR	R23@>54, ZR, R14; \
	MOVD	R15, R23; \
	ORR	R22@>49, ZR, R15; \
	MOVD	R3, R22; \
	ORR	R23@>43, ZR, R3; \
	MOVD	R5, R23; \
	ORR	R22@>36, ZR, R5; \
	MOVD	48(RSP), R22; \
	ORR	R23@>28, ZR, R24; \
	MOVD	R24, 48(RSP); \
	MOVD	R7, R23; \
	ORR	R22@>19, ZR, R7; \
	MOVD	56(RSP), R22; \
	ORR	R23@>9, ZR, R24; \
	MOVD	R24, 56(RSP); \
	MOVD	R21, R23; \
	ORR	R22@>62, ZR, R21; \
	MOVD	R4, R22; \
	ORR	R23@>50, ZR, R4; \
	MOVD	R13, R23; \
	ORR	R22@>37, ZR, R13; \
	MOVD	R20, R22; \
	ORR	R23@>23, ZR, R20; \
	MOVD	R16, R23; \
	ORR	R22@>8, ZR, R16; \
	MOVD	R11, R22; \
	ORR	R23@>56, ZR, R11; \
	MOVD	R10, R23; \
	ORR	R22@>39, ZR, R10; \
	MOVD	R2, R22; \
	ORR	R23@>21, ZR, R2; \
	MOVD	R17, R23; \
	ORR	R22@>2, ZR, R17; \
	MOVD	R12, R22; \
	ORR	R23@>46, ZR, R12; \
	MOVD	R19, R23; \
	ORR	R22@>25, ZR, R19; \
	MOVD	R8, R22; \
	ORR	R23@>3, ZR, R8; \
	MOVD	32(RSP), R23; \
	ORR	R22@>44, ZR, R24; \
	MOVD	R24, 32(RSP); \
	ORR	R23@>20, ZR, R24; \
	MOVD	R24, 64(RSP); \
	/* chi row 0: R0, 64(RSP), R2, R3, R4 */ \
	MOVD	R0, R22; \
	MOVD	64(RSP), R23; \
	BIC	R23, R2, R0; \
	EOR	R22, R0, R0; \
	BIC	R2, R3, R24; \
	EOR	R23, R24, R24; \
	MOVD	R24, 64(RSP); \
	BIC	R3, R4, R24; \
	EOR	R24, R2, R2; \
	BIC	R4, R22, R24; \
	EOR	R24, R3, R3; \
	BIC	R22, R23, R24; \
	EOR	R24, R4, R4; \
	/* chi row 1: R5, 32(RSP), R6, R7, R8 */ \
	MOVD	R5, R22; \
	MOVD	32(RSP), R23; \
	BIC	R23, R6, R5; \
	EOR	R22, R5, R5; \
	BIC	R6, R7, R24; \
	EOR	R23, R24, R24; \
	MOVD	R24, 32(RSP); \
	BIC	R7, R8, R24; \
	EOR	R24, R6, R6; \
	BIC	R8, R22, R24; \
	EOR	R24, R7, R7; \
	BIC	R22, R23, R24; \
	EOR	R24, R8, R8; \
	/* chi row 2: R9, 40(RSP), R10, R11, R12 */ \
	MOVD	R9, R22; \
	MOVD	40(RSP), R23; \
	BIC	R23, R10, R9; \
	EOR	R22, R9, R9; \
	BIC	R10, R11, R24; \
	EOR	R23, R24, R24; \
	MOVD	R24, 40(RSP); \
	BIC	R11, R12, R24; \
	EOR	R24, R10, R10; \
	BIC	R12, R22, R24; \
	EOR	R24, R11, R11; \
	BIC	R22, R23, R24; \
	EOR	R24, R12, R12; \
	/* chi row 3: R13, 48(RSP), R14, R15, R16 */ \
	MOVD	R13, R22; \
	MOVD	48(RSP), R23; \
	BIC	R23, R14, R13; \
	EOR	R22, R13, R13; \
	BIC	R14, R15, R24; \
	EOR	R23, R24, R24; \
	MOVD	R24, 48(RSP); \
	BIC	R15, R16, R24; \
	EOR	R24, R14, R14; \
	BIC	R16, R22, R24; \
	EOR	R24, R15, R15; \
	BIC	R22, R23, R24; \
	EOR	R24, R16, R16; \
	/* chi row 4: R17, 56(RSP), R19, R20, R21 */ \
	MOVD	R17, R22; \
	MOVD	56(RSP), R23; \
	BIC	R23, R19, R17; \
	EOR	R22, R17, R17; \
	BIC	R19, R20, R24; \
	EOR	R23, R24, R24; \
	MOVD	R24, 56(RSP); \
	BIC	R20, R21, R24; \
	EOR	R24, R19, R19; \
	BIC	R21, R22, R24; \
	EOR	R24, R20, R20; \
	BIC	R22, R23, R24; \
	EOR	R24, R21, R21; \
	/* iota */ \
	MOVD	rc, R22; \
	EOR	R22, R0, R0

// SCALAR_ABSORB_LANE XORs one 8-byte input lane into a register-resident
// state lane; the source pointer walks in R25. SCALAR_ABSORB_LANE_MEM is the
// spilled-lane variant.
#define SCALAR_ABSORB_LANE(A) \
	MOVD.P	8(R25), R22; \
	EOR	R22, A, A

#define SCALAR_ABSORB_LANE_MEM(off) \
	MOVD.P	8(R25), R22; \
	MOVD	off(RSP), R24; \
	EOR	R22, R24, R24; \
	MOVD	R24, off(RSP)

// SCALAR_ABSORB_LANES21 absorbs one full 168-byte rate block (lanes 0-20)
// in byte order.
#define SCALAR_ABSORB_LANES21 \
	SCALAR_ABSORB_LANE(R0); \
	SCALAR_ABSORB_LANE_MEM(64); \
	SCALAR_ABSORB_LANE(R2); \
	SCALAR_ABSORB_LANE(R3); \
	SCALAR_ABSORB_LANE(R4); \
	SCALAR_ABSORB_LANE(R5); \
	SCALAR_ABSORB_LANE_MEM(32); \
	SCALAR_ABSORB_LANE(R6); \
	SCALAR_ABSORB_LANE(R7); \
	SCALAR_ABSORB_LANE(R8); \
	SCALAR_ABSORB_LANE(R9); \
	SCALAR_ABSORB_LANE_MEM(40); \
	SCALAR_ABSORB_LANE(R10); \
	SCALAR_ABSORB_LANE(R11); \
	SCALAR_ABSORB_LANE(R12); \
	SCALAR_ABSORB_LANE(R13); \
	SCALAR_ABSORB_LANE_MEM(48); \
	SCALAR_ABSORB_LANE(R14); \
	SCALAR_ABSORB_LANE(R15); \
	SCALAR_ABSORB_LANE(R16); \
	SCALAR_ABSORB_LANE(R17)

// SCALAR_ABSORB_TAIL16_LAST absorbs the final 128-byte partial block of a
// KT128 leaf (lanes 0-15), then XORs the leaf padding: DS 0x0B into the low
// byte of lane 16 and the pad10*1 end bit into the top byte of lane 20.
#define SCALAR_ABSORB_TAIL16_LAST \
	SCALAR_ABSORB_LANE(R0); \
	SCALAR_ABSORB_LANE_MEM(64); \
	SCALAR_ABSORB_LANE(R2); \
	SCALAR_ABSORB_LANE(R3); \
	SCALAR_ABSORB_LANE(R4); \
	SCALAR_ABSORB_LANE(R5); \
	SCALAR_ABSORB_LANE_MEM(32); \
	SCALAR_ABSORB_LANE(R6); \
	SCALAR_ABSORB_LANE(R7); \
	SCALAR_ABSORB_LANE(R8); \
	SCALAR_ABSORB_LANE(R9); \
	SCALAR_ABSORB_LANE_MEM(40); \
	SCALAR_ABSORB_LANE(R10); \
	SCALAR_ABSORB_LANE(R11); \
	SCALAR_ABSORB_LANE(R12); \
	SCALAR_ABSORB_LANE(R13); \
	MOVD	$0x0B, R22; \
	MOVD	48(RSP), R24; \
	EOR	R22, R24, R24; \
	MOVD	R24, 48(RSP); \
	MOVD	$0x8000000000000000, R22; \
	EOR	R22, R17, R17

// x4 AVX-512VL Keccak permutation macros: the x8 lane-major structure at YMM
// width. State lives in Y0-Y24 (25 lanes × 4 uint64s per YMM register);
// Y25-Y31 are scratch. All instructions are the 256-bit EVEX forms of the x8
// kernel's, so this requires AVX-512F+VL (implied by the package's HasAVX512
// gate). R11 must point to the round constant table.
//
// KEEP IN SYNC with the X8_* macros in permute_amd64_avx512.h and the X2_*
// macros in permute_amd64_x2.h: the Go asm preprocessor cannot parameterize
// the register class, so the three are parallel transcriptions of the same
// schedule (register-for-register, Z<n> ↔ Y<n> ↔ X<n>).

// XOR five lanes into dst: dst = a ^ b ^ c ^ d ^ e.
#define XOR5_AVX512_4X(dst, a, b, c, d, e) \
	VMOVDQU64	a, dst; \
	VPTERNLOGQ	$0x96, c, b, dst; \
	VPTERNLOGQ	$0x96, e, d, dst

// Theta in-place: compute column parities C[0..4] in Y25-Y29, then apply
// D[x] = C[(x+4)%5] ^ ROT(C[(x+1)%5],1) directly into each state lane using
// VPTERNLOGQ to fuse the D formation with the state XOR.
#define X4_THETA_INPLACE() \
	XOR5_AVX512_4X(Y25, Y0, Y5, Y10, Y15, Y20); \
	XOR5_AVX512_4X(Y26, Y1, Y6, Y11, Y16, Y21); \
	XOR5_AVX512_4X(Y27, Y2, Y7, Y12, Y17, Y22); \
	XOR5_AVX512_4X(Y28, Y3, Y8, Y13, Y18, Y23); \
	XOR5_AVX512_4X(Y29, Y4, Y9, Y14, Y19, Y24); \
	/* D[0] = C[4] ^ ROT(C[1],1) — column 0 */ \
	VPROLQ	$1, Y26, Y30; \
	VPTERNLOGQ	$0x96, Y29, Y30, Y0; \
	VPTERNLOGQ	$0x96, Y29, Y30, Y5; \
	VPTERNLOGQ	$0x96, Y29, Y30, Y10; \
	VPTERNLOGQ	$0x96, Y29, Y30, Y15; \
	VPTERNLOGQ	$0x96, Y29, Y30, Y20; \
	/* D[1] = C[0] ^ ROT(C[2],1) — column 1 */ \
	VPROLQ	$1, Y27, Y30; \
	VPTERNLOGQ	$0x96, Y25, Y30, Y1; \
	VPTERNLOGQ	$0x96, Y25, Y30, Y6; \
	VPTERNLOGQ	$0x96, Y25, Y30, Y11; \
	VPTERNLOGQ	$0x96, Y25, Y30, Y16; \
	VPTERNLOGQ	$0x96, Y25, Y30, Y21; \
	/* D[2] = C[1] ^ ROT(C[3],1) — column 2 */ \
	VPROLQ	$1, Y28, Y30; \
	VPTERNLOGQ	$0x96, Y26, Y30, Y2; \
	VPTERNLOGQ	$0x96, Y26, Y30, Y7; \
	VPTERNLOGQ	$0x96, Y26, Y30, Y12; \
	VPTERNLOGQ	$0x96, Y26, Y30, Y17; \
	VPTERNLOGQ	$0x96, Y26, Y30, Y22; \
	/* D[3] = C[2] ^ ROT(C[4],1) — column 3 */ \
	VPROLQ	$1, Y29, Y30; \
	VPTERNLOGQ	$0x96, Y27, Y30, Y3; \
	VPTERNLOGQ	$0x96, Y27, Y30, Y8; \
	VPTERNLOGQ	$0x96, Y27, Y30, Y13; \
	VPTERNLOGQ	$0x96, Y27, Y30, Y18; \
	VPTERNLOGQ	$0x96, Y27, Y30, Y23; \
	/* D[4] = C[3] ^ ROT(C[0],1) — column 4 */ \
	VPROLQ	$1, Y25, Y30; \
	VPTERNLOGQ	$0x96, Y28, Y30, Y4; \
	VPTERNLOGQ	$0x96, Y28, Y30, Y9; \
	VPTERNLOGQ	$0x96, Y28, Y30, Y14; \
	VPTERNLOGQ	$0x96, Y28, Y30, Y19; \
	VPTERNLOGQ	$0x96, Y28, Y30, Y24

// Rho/Pi/Chi row transform (theta already applied in-place).
#define X4_RPC_MAP(L1, L2, L3, L4, L5, R1, R2, R3, R4, R5, A, B, C, D, E) \
	VPROLQ	$R1, L1, Y25; \
	VPROLQ	$R2, L2, Y26; \
	VPROLQ	$R3, L3, Y27; \
	VPROLQ	$R4, L4, Y28; \
	VPROLQ	$R5, L5, Y29; \
	VMOVDQU64	A, Y30; \
	VMOVDQU64	B, Y31; \
	VPTERNLOGQ	$0xD2, C, B, A; \
	VPTERNLOGQ	$0xD2, D, C, B; \
	VPTERNLOGQ	$0xD2, E, D, C; \
	VPTERNLOGQ	$0xD2, Y30, E, D; \
	VPTERNLOGQ	$0xD2, Y31, Y30, E; \
	VMOVDQU64	A, L1; \
	VMOVDQU64	B, L2; \
	VMOVDQU64	C, L3; \
	VMOVDQU64	D, L4; \
	VMOVDQU64	E, L5

// Same as X4_RPC_MAP, but xor round constant into lane A after Chi (Iota).
#define X4_RPC_IOTA_MAP(L1, L2, L3, L4, L5, R1, R2, R3, R4, R5, A, B, C, D, E, RC_OFF) \
	VPROLQ	$R1, L1, Y25; \
	VPROLQ	$R2, L2, Y26; \
	VPROLQ	$R3, L3, Y27; \
	VPROLQ	$R4, L4, Y28; \
	VPROLQ	$R5, L5, Y29; \
	VMOVDQU64	A, Y30; \
	VMOVDQU64	B, Y31; \
	VPTERNLOGQ	$0xD2, C, B, A; \
	VPTERNLOGQ	$0xD2, D, C, B; \
	VPTERNLOGQ	$0xD2, E, D, C; \
	VPTERNLOGQ	$0xD2, Y30, E, D; \
	VPTERNLOGQ	$0xD2, Y31, Y30, E; \
	VPBROADCASTQ	RC_OFF(R11), Y30; \
	VPXORQ	Y30, A, A; \
	VMOVDQU64	A, L1; \
	VMOVDQU64	B, L2; \
	VMOVDQU64	C, L3; \
	VMOVDQU64	D, L4; \
	VMOVDQU64	E, L5

// Four unrolled rounds in the exact XKCP lane schedule.
#define X4_4ROUNDS(off0, off1, off2, off3) \
	X4_THETA_INPLACE(); \
	X4_RPC_IOTA_MAP(Y0, Y6, Y12, Y18, Y24, 0, 44, 43, 21, 14, Y25, Y26, Y27, Y28, Y29, off0); \
	X4_RPC_MAP(Y10, Y16, Y22, Y3, Y9, 3, 45, 61, 28, 20, Y28, Y29, Y25, Y26, Y27); \
	X4_RPC_MAP(Y20, Y1, Y7, Y13, Y19, 18, 1, 6, 25, 8, Y26, Y27, Y28, Y29, Y25); \
	X4_RPC_MAP(Y5, Y11, Y17, Y23, Y4, 36, 10, 15, 56, 27, Y29, Y25, Y26, Y27, Y28); \
	X4_RPC_MAP(Y15, Y21, Y2, Y8, Y14, 41, 2, 62, 55, 39, Y27, Y28, Y29, Y25, Y26); \
	\
	X4_THETA_INPLACE(); \
	X4_RPC_IOTA_MAP(Y0, Y16, Y7, Y23, Y14, 0, 44, 43, 21, 14, Y25, Y26, Y27, Y28, Y29, off1); \
	X4_RPC_MAP(Y20, Y11, Y2, Y18, Y9, 3, 45, 61, 28, 20, Y28, Y29, Y25, Y26, Y27); \
	X4_RPC_MAP(Y15, Y6, Y22, Y13, Y4, 18, 1, 6, 25, 8, Y26, Y27, Y28, Y29, Y25); \
	X4_RPC_MAP(Y10, Y1, Y17, Y8, Y24, 36, 10, 15, 56, 27, Y29, Y25, Y26, Y27, Y28); \
	X4_RPC_MAP(Y5, Y21, Y12, Y3, Y19, 41, 2, 62, 55, 39, Y27, Y28, Y29, Y25, Y26); \
	\
	X4_THETA_INPLACE(); \
	X4_RPC_IOTA_MAP(Y0, Y11, Y22, Y8, Y19, 0, 44, 43, 21, 14, Y25, Y26, Y27, Y28, Y29, off2); \
	X4_RPC_MAP(Y15, Y1, Y12, Y23, Y9, 3, 45, 61, 28, 20, Y28, Y29, Y25, Y26, Y27); \
	X4_RPC_MAP(Y5, Y16, Y2, Y13, Y24, 18, 1, 6, 25, 8, Y26, Y27, Y28, Y29, Y25); \
	X4_RPC_MAP(Y20, Y6, Y17, Y3, Y14, 36, 10, 15, 56, 27, Y29, Y25, Y26, Y27, Y28); \
	X4_RPC_MAP(Y10, Y21, Y7, Y18, Y4, 41, 2, 62, 55, 39, Y27, Y28, Y29, Y25, Y26); \
	\
	X4_THETA_INPLACE(); \
	X4_RPC_IOTA_MAP(Y0, Y1, Y2, Y3, Y4, 0, 44, 43, 21, 14, Y25, Y26, Y27, Y28, Y29, off3); \
	X4_RPC_MAP(Y5, Y6, Y7, Y8, Y9, 3, 45, 61, 28, 20, Y28, Y29, Y25, Y26, Y27); \
	X4_RPC_MAP(Y10, Y11, Y12, Y13, Y14, 18, 1, 6, 25, 8, Y26, Y27, Y28, Y29, Y25); \
	X4_RPC_MAP(Y15, Y16, Y17, Y18, Y19, 36, 10, 15, 56, 27, Y29, Y25, Y26, Y27, Y28); \
	X4_RPC_MAP(Y20, Y21, Y22, Y23, Y24, 41, 2, 62, 55, 39, Y27, Y28, Y29, Y25, Y26)

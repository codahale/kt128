// x2 AVX-512VL Keccak permutation macros: the x8 lane-major structure at XMM
// width. State lives in X0-X24 (25 lanes × 2 uint64s per XMM register);
// X25-X31 are scratch. All instructions are the 128-bit EVEX forms of the x8
// kernel's, so this requires AVX-512F+VL (implied by the package's HasAVX512
// gate). R11 must point to the round constant table.

// XOR five lanes into dst: dst = a ^ b ^ c ^ d ^ e.
#define XOR5_AVX512_2X(dst, a, b, c, d, e) \
	VMOVDQU64	a, dst; \
	VPTERNLOGQ	$0x96, c, b, dst; \
	VPTERNLOGQ	$0x96, e, d, dst

// Theta in-place: compute column parities C[0..4] in X25-X29, then apply
// D[x] = C[(x+4)%5] ^ ROT(C[(x+1)%5],1) directly into each state lane using
// VPTERNLOGQ to fuse the D formation with the state XOR.
#define X2_THETA_INPLACE() \
	XOR5_AVX512_2X(X25, X0, X5, X10, X15, X20); \
	XOR5_AVX512_2X(X26, X1, X6, X11, X16, X21); \
	XOR5_AVX512_2X(X27, X2, X7, X12, X17, X22); \
	XOR5_AVX512_2X(X28, X3, X8, X13, X18, X23); \
	XOR5_AVX512_2X(X29, X4, X9, X14, X19, X24); \
	/* D[0] = C[4] ^ ROT(C[1],1) — column 0 */ \
	VPROLQ	$1, X26, X30; \
	VPTERNLOGQ	$0x96, X29, X30, X0; \
	VPTERNLOGQ	$0x96, X29, X30, X5; \
	VPTERNLOGQ	$0x96, X29, X30, X10; \
	VPTERNLOGQ	$0x96, X29, X30, X15; \
	VPTERNLOGQ	$0x96, X29, X30, X20; \
	/* D[1] = C[0] ^ ROT(C[2],1) — column 1 */ \
	VPROLQ	$1, X27, X30; \
	VPTERNLOGQ	$0x96, X25, X30, X1; \
	VPTERNLOGQ	$0x96, X25, X30, X6; \
	VPTERNLOGQ	$0x96, X25, X30, X11; \
	VPTERNLOGQ	$0x96, X25, X30, X16; \
	VPTERNLOGQ	$0x96, X25, X30, X21; \
	/* D[2] = C[1] ^ ROT(C[3],1) — column 2 */ \
	VPROLQ	$1, X28, X30; \
	VPTERNLOGQ	$0x96, X26, X30, X2; \
	VPTERNLOGQ	$0x96, X26, X30, X7; \
	VPTERNLOGQ	$0x96, X26, X30, X12; \
	VPTERNLOGQ	$0x96, X26, X30, X17; \
	VPTERNLOGQ	$0x96, X26, X30, X22; \
	/* D[3] = C[2] ^ ROT(C[4],1) — column 3 */ \
	VPROLQ	$1, X29, X30; \
	VPTERNLOGQ	$0x96, X27, X30, X3; \
	VPTERNLOGQ	$0x96, X27, X30, X8; \
	VPTERNLOGQ	$0x96, X27, X30, X13; \
	VPTERNLOGQ	$0x96, X27, X30, X18; \
	VPTERNLOGQ	$0x96, X27, X30, X23; \
	/* D[4] = C[3] ^ ROT(C[0],1) — column 4 */ \
	VPROLQ	$1, X25, X30; \
	VPTERNLOGQ	$0x96, X28, X30, X4; \
	VPTERNLOGQ	$0x96, X28, X30, X9; \
	VPTERNLOGQ	$0x96, X28, X30, X14; \
	VPTERNLOGQ	$0x96, X28, X30, X19; \
	VPTERNLOGQ	$0x96, X28, X30, X24

// Rho/Pi/Chi row transform (theta already applied in-place).
#define X2_RPC_MAP(L1, L2, L3, L4, L5, R1, R2, R3, R4, R5, A, B, C, D, E) \
	VPROLQ	$R1, L1, X25; \
	VPROLQ	$R2, L2, X26; \
	VPROLQ	$R3, L3, X27; \
	VPROLQ	$R4, L4, X28; \
	VPROLQ	$R5, L5, X29; \
	VMOVDQU64	A, X30; \
	VMOVDQU64	B, X31; \
	VPTERNLOGQ	$0xD2, C, B, A; \
	VPTERNLOGQ	$0xD2, D, C, B; \
	VPTERNLOGQ	$0xD2, E, D, C; \
	VPTERNLOGQ	$0xD2, X30, E, D; \
	VPTERNLOGQ	$0xD2, X31, X30, E; \
	VMOVDQU64	A, L1; \
	VMOVDQU64	B, L2; \
	VMOVDQU64	C, L3; \
	VMOVDQU64	D, L4; \
	VMOVDQU64	E, L5

// Same as X2_RPC_MAP, but xor round constant into lane A after Chi (Iota).
#define X2_RPC_IOTA_MAP(L1, L2, L3, L4, L5, R1, R2, R3, R4, R5, A, B, C, D, E, RC_OFF) \
	VPROLQ	$R1, L1, X25; \
	VPROLQ	$R2, L2, X26; \
	VPROLQ	$R3, L3, X27; \
	VPROLQ	$R4, L4, X28; \
	VPROLQ	$R5, L5, X29; \
	VMOVDQU64	A, X30; \
	VMOVDQU64	B, X31; \
	VPTERNLOGQ	$0xD2, C, B, A; \
	VPTERNLOGQ	$0xD2, D, C, B; \
	VPTERNLOGQ	$0xD2, E, D, C; \
	VPTERNLOGQ	$0xD2, X30, E, D; \
	VPTERNLOGQ	$0xD2, X31, X30, E; \
	VPBROADCASTQ	RC_OFF(R11), X30; \
	VPXORQ	X30, A, A; \
	VMOVDQU64	A, L1; \
	VMOVDQU64	B, L2; \
	VMOVDQU64	C, L3; \
	VMOVDQU64	D, L4; \
	VMOVDQU64	E, L5

// Four unrolled rounds in the exact XKCP lane schedule.
#define X2_4ROUNDS(off0, off1, off2, off3) \
	X2_THETA_INPLACE(); \
	X2_RPC_IOTA_MAP(X0, X6, X12, X18, X24, 0, 44, 43, 21, 14, X25, X26, X27, X28, X29, off0); \
	X2_RPC_MAP(X10, X16, X22, X3, X9, 3, 45, 61, 28, 20, X28, X29, X25, X26, X27); \
	X2_RPC_MAP(X20, X1, X7, X13, X19, 18, 1, 6, 25, 8, X26, X27, X28, X29, X25); \
	X2_RPC_MAP(X5, X11, X17, X23, X4, 36, 10, 15, 56, 27, X29, X25, X26, X27, X28); \
	X2_RPC_MAP(X15, X21, X2, X8, X14, 41, 2, 62, 55, 39, X27, X28, X29, X25, X26); \
	\
	X2_THETA_INPLACE(); \
	X2_RPC_IOTA_MAP(X0, X16, X7, X23, X14, 0, 44, 43, 21, 14, X25, X26, X27, X28, X29, off1); \
	X2_RPC_MAP(X20, X11, X2, X18, X9, 3, 45, 61, 28, 20, X28, X29, X25, X26, X27); \
	X2_RPC_MAP(X15, X6, X22, X13, X4, 18, 1, 6, 25, 8, X26, X27, X28, X29, X25); \
	X2_RPC_MAP(X10, X1, X17, X8, X24, 36, 10, 15, 56, 27, X29, X25, X26, X27, X28); \
	X2_RPC_MAP(X5, X21, X12, X3, X19, 41, 2, 62, 55, 39, X27, X28, X29, X25, X26); \
	\
	X2_THETA_INPLACE(); \
	X2_RPC_IOTA_MAP(X0, X11, X22, X8, X19, 0, 44, 43, 21, 14, X25, X26, X27, X28, X29, off2); \
	X2_RPC_MAP(X15, X1, X12, X23, X9, 3, 45, 61, 28, 20, X28, X29, X25, X26, X27); \
	X2_RPC_MAP(X5, X16, X2, X13, X24, 18, 1, 6, 25, 8, X26, X27, X28, X29, X25); \
	X2_RPC_MAP(X20, X6, X17, X3, X14, 36, 10, 15, 56, 27, X29, X25, X26, X27, X28); \
	X2_RPC_MAP(X10, X21, X7, X18, X4, 41, 2, 62, 55, 39, X27, X28, X29, X25, X26); \
	\
	X2_THETA_INPLACE(); \
	X2_RPC_IOTA_MAP(X0, X1, X2, X3, X4, 0, 44, 43, 21, 14, X25, X26, X27, X28, X29, off3); \
	X2_RPC_MAP(X5, X6, X7, X8, X9, 3, 45, 61, 28, 20, X28, X29, X25, X26, X27); \
	X2_RPC_MAP(X10, X11, X12, X13, X14, 18, 1, 6, 25, 8, X26, X27, X28, X29, X25); \
	X2_RPC_MAP(X15, X16, X17, X18, X19, 36, 10, 15, 56, 27, X29, X25, X26, X27, X28); \
	X2_RPC_MAP(X20, X21, X22, X23, X24, 41, 2, 62, 55, 39, X27, X28, X29, X25, X26)

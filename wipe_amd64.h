#define WIPE_STACK_QWORDS(n) \
	XORQ AX, AX; \
	MOVQ SP, DI; \
	MOVQ $n, CX; \
	REP; \
	STOSQ

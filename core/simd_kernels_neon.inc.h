// simd_kernels_neon.inc.h instantiates the kernel bodies in simd_kernels.inc.h for NEON on arm64: 128-bit vectors,
// four independent accumulators per chain, and a fused HANN_FMA. NEON is a baseline feature of aarch64, so the
// kernels need no per-function target attribute and no runtime CPU check beyond the architecture itself, which is
// why HANN_TARGET expands to nothing. Note the argument order of vfmaq_f32: the accumulator comes first, the two
// factors after it, which is the opposite of _mm256_fmadd_ps.
#define HANN_SUFFIX neon
#define HANN_TARGET
#define HANN_UNROLL 4
#define HANN_VEC float32x4_t
#define HANN_LANES 4
#define HANN_ZERO() vdupq_n_f32(0.0f)
#define HANN_LOAD(p) vld1q_f32(p)
#define HANN_STORE(p, v) vst1q_f32((p), (v))
#define HANN_SET1(x) vdupq_n_f32(x)
#define HANN_ADD(x, y) vaddq_f32((x), (y))
#define HANN_SUB(x, y) vsubq_f32((x), (y))
#define HANN_MUL(x, y) vmulq_f32((x), (y))
#define HANN_DIV(x, y) vdivq_f32((x), (y))
#define HANN_FMA(x, y, acc) vfmaq_f32((acc), (x), (y))
#define HANN_ABS(v) vabsq_f32(v)
#define HANN_REDUCE(v) vaddvq_f32(v)

#include "simd_kernels.inc.h"

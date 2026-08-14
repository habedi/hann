// simd_kernels_avx.inc.h instantiates the kernel bodies in simd_kernels.inc.h for the AVX tier: 256-bit vectors,
// one accumulator per chain, and a HANN_FMA that expands to a multiply followed by an add, because plain AVX has
// no fused multiply-add instruction.
#define HANN_SUFFIX avx
#define HANN_TARGET HANN_TARGET_AVX
#define HANN_UNROLL 1
#define HANN_VEC __m256
#define HANN_LANES 8
#define HANN_ZERO() _mm256_setzero_ps()
#define HANN_LOAD(p) _mm256_loadu_ps(p)
#define HANN_STORE(p, v) _mm256_storeu_ps((p), (v))
#define HANN_SET1(x) _mm256_set1_ps(x)
#define HANN_ADD(x, y) _mm256_add_ps((x), (y))
#define HANN_SUB(x, y) _mm256_sub_ps((x), (y))
#define HANN_MUL(x, y) _mm256_mul_ps((x), (y))
#define HANN_DIV(x, y) _mm256_div_ps((x), (y))
#define HANN_FMA(x, y, acc) HANN_ADD((acc), HANN_MUL((x), (y)))
#define HANN_ABS(v) _mm256_andnot_ps(_mm256_set1_ps(-0.0f), (v))
#define HANN_REDUCE(v) horizontal_sum256(v)

#include "simd_kernels.inc.h"

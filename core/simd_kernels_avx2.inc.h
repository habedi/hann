// simd_kernels_avx2.inc.h instantiates the kernel bodies in simd_kernels.inc.h for the AVX2 tier: 256-bit vectors,
// four independent accumulators per chain, and a fused HANN_FMA. The target attribute enables both avx2 and fma,
// and hann_cpu_init selects this tier only when the CPU reports both features.
#define HANN_SUFFIX avx2
#define HANN_TARGET HANN_TARGET_AVX2
#define HANN_UNROLL 4
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
#define HANN_FMA(x, y, acc) _mm256_fmadd_ps((x), (y), (acc))
#define HANN_ABS(v) _mm256_andnot_ps(_mm256_set1_ps(-0.0f), (v))
#define HANN_REDUCE(v) horizontal_sum256(v)

#include "simd_kernels.inc.h"

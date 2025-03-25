#include "simd_ops.h"
#include <immintrin.h>
#include <math.h>
#include <stddef.h>

/**
 * @brief Computes the horizontal sum of a 256-bit vector.
 *
 * @param v The 256-bit vector.
 * @return The sum of all elements in the vector.
 */
static inline float horizontal_sum256(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/**
 * @brief Normalizes a vector using AVX instructions.
 *
 * This function normalizes the input vector by dividing each element by the vector's norm.
 *
 * @param vec Pointer to the vector to be normalized.
 * @param len The number of elements in the vector.
 */
void avx_normalize(float *vec, size_t len) {
    __m256 sum = _mm256_setzero_ps();
    size_t i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
#ifdef __FMA__
        sum = _mm256_fmadd_ps(v, v, sum);
#else
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v, v));
#endif
    }
    float total = horizontal_sum256(sum);
    for (; i < len; i++) {
        total += vec[i] * vec[i];
    }
    float norm = sqrtf(total);
    if (norm == 0.0f) return;
    __m256 norm_vec = _mm256_set1_ps(norm);
    for (i = 0; i <= len - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        v = _mm256_div_ps(v, norm_vec);
        _mm256_storeu_ps(&vec[i], v);
    }
    for (; i < len; i++) {
        vec[i] /= norm;
    }
}

#include "simd_ops.h"
#include <immintrin.h>
#include <math.h>

/**
 * \brief Normalizes a single vector using AVX instructions.
 *
 * This function normalizes a vector of floats by dividing each element by the
 * vector's Euclidean norm.
 *
 * \param vec Pointer to the vector to be normalized.
 * \param len Length of the vector.
 */
void avx_normalize(float *vec, int len) {
    __m256 sum = _mm256_setzero_ps();

    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v, v));
    }

    float sums[8];
    _mm256_storeu_ps(sums, sum);

    float total = 0.0f;
    for (int k = 0; k < 8; k++) {
        total += sums[k];
    }

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

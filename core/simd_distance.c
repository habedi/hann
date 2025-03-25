#include "simd_distance.h"
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
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/**
 * @brief Computes the Euclidean distance between two vectors using SIMD instructions.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param n The number of elements in the vectors.
 * @return The Euclidean distance between the two vectors.
 */
float simd_euclidean(const float* a, const float* b, size_t n) {
    if (!a || !b) return NAN;

    size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    size_t limit = n - (n % 8);

    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum_vec = _mm256_add_ps(sum_vec, sq);
    }

    float sum = horizontal_sum256(sum_vec);

    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/**
 * @brief Computes the squared Euclidean distance between two vectors using SIMD instructions.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param n The number of elements in the vectors.
 * @return The squared Euclidean distance between the two vectors.
 */
float simd_squared_euclidean(const float* a, const float* b, size_t n) {
    if (!a || !b) return NAN;

    size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    size_t limit = n - (n % 8);

    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum_vec = _mm256_add_ps(sum_vec, sq);
    }

    float sum = horizontal_sum256(sum_vec);

    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * @brief Computes the Manhattan distance between two vectors using SIMD instructions.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param n The number of elements in the vectors.
 * @return The Manhattan distance between the two vectors.
 */
float simd_manhattan(const float* a, const float* b, size_t n) {
    if (!a || !b) return NAN;

    size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    size_t limit = n - (n % 8);

    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 abs_diff = _mm256_andnot_ps(sign_mask, diff);
        sum_vec = _mm256_add_ps(sum_vec, abs_diff);
    }

    float sum = horizontal_sum256(sum_vec);

    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff < 0 ? -diff : diff;
    }
    return sum;
}

/**
 * @brief Computes the cosine distance between two vectors using SIMD instructions.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param n The number of elements in the vectors.
 * @return The cosine distance between the two vectors.
 */
float simd_cosine_distance(const float* a, const float* b, size_t n) {
    if (!a || !b) return NAN;

    size_t i = 0;
    __m256 dot_vec = _mm256_setzero_ps();
    __m256 norm_a_vec = _mm256_setzero_ps();
    __m256 norm_b_vec = _mm256_setzero_ps();
    size_t limit = n - (n % 8);

    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#ifdef __FMA__
        dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
        norm_a_vec = _mm256_fmadd_ps(va, va, norm_a_vec);
        norm_b_vec = _mm256_fmadd_ps(vb, vb, norm_b_vec);
#else
        dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va, vb));
        norm_a_vec = _mm256_add_ps(norm_a_vec, _mm256_mul_ps(va, va));
        norm_b_vec = _mm256_add_ps(norm_b_vec, _mm256_mul_ps(vb, vb));
#endif
    }

    float dot = horizontal_sum256(dot_vec);
    float norm_a = horizontal_sum256(norm_a_vec);
    float norm_b = horizontal_sum256(norm_b_vec);

    for (; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float normA = sqrtf(norm_a);
    float normB = sqrtf(norm_b);
    if (normA == 0.0f || normB == 0.0f) {
        return 1.0f;
    }

    float cosine_similarity = dot / (normA * normB);
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;

    return 1.0f - cosine_similarity;
}

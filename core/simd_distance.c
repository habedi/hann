#include "simd_distance.h"
#include <immintrin.h>
#include <math.h>

/**
 * \brief Computes the Euclidean distance between two vectors using SIMD instructions.
 *
 * \param a Pointer to the first vector.
 * \param b Pointer to the second vector.
 * \param n The number of elements in the vectors.
 * \return The Euclidean distance between the two vectors.
 */
float simd_euclidean(const float* a, const float* b, size_t n) {
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
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/**
 * \brief Computes the squared Euclidean distance between two vectors using SIMD instructions.
 *
 * \param a Pointer to the first vector.
 * \param b Pointer to the second vector.
 * \param n The number of elements in the vectors.
 * \return The squared Euclidean distance between the two vectors.
 */
float simd_squared_euclidean(const float* a, const float* b, size_t n) {
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
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * \brief Computes the Manhattan distance between two vectors using SIMD instructions.
 *
 * \param a Pointer to the first vector.
 * \param b Pointer to the second vector.
 * \param n The number of elements in the vectors.
 * \return The Manhattan distance between the two vectors.
 */
float simd_manhattan(const float* a, const float* b, size_t n) {
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
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff < 0 ? -diff : diff;
    }
    return sum;
}

/**
 * \brief Computes the cosine distance between two vectors using SIMD instructions.
 *
 * \param a Pointer to the first vector.
 * \param b Pointer to the second vector.
 * \param n The number of elements in the vectors.
 * \return The cosine distance between the two vectors.
 */
float simd_cosine_distance(const float* a, const float* b, size_t n) {
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
    float dot_array[8], norm_a_array[8], norm_b_array[8];
    _mm256_storeu_ps(dot_array, dot_vec);
    _mm256_storeu_ps(norm_a_array, norm_a_vec);
    _mm256_storeu_ps(norm_b_array, norm_b_vec);
    float dot = dot_array[0] + dot_array[1] + dot_array[2] + dot_array[3] +
                dot_array[4] + dot_array[5] + dot_array[6] + dot_array[7];
    float norm_a = norm_a_array[0] + norm_a_array[1] + norm_a_array[2] + norm_a_array[3] +
                   norm_a_array[4] + norm_a_array[5] + norm_a_array[6] + norm_a_array[7];
    float norm_b = norm_b_array[0] + norm_b_array[1] + norm_b_array[2] + norm_b_array[3] +
                   norm_b_array[4] + norm_b_array[5] + norm_b_array[6] + norm_b_array[7];
    for (; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float normA = sqrtf(norm_a);
    float normB = sqrtf(norm_b);
    if (normA == 0.0f || normB == 0.0f) {
        return 1.0f; // fallback: cosine distance of 1
    }
    float cosine_similarity = dot / (normA * normB);
    return 1.0f - cosine_similarity;
}

/**
 * \brief Computes the angular distance between two vectors using SIMD instructions.
 *
 * \param a Pointer to the first vector.
 * \param b Pointer to the second vector.
 * \param n The number of elements in the vectors.
 * \return The angular distance between the two vectors.
 */
float simd_angular_distance(const float* a, const float* b, size_t n) {
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
    float dot_array[8], norm_a_array[8], norm_b_array[8];
    _mm256_storeu_ps(dot_array, dot_vec);
    _mm256_storeu_ps(norm_a_array, norm_a_vec);
    _mm256_storeu_ps(norm_b_array, norm_b_vec);
    float dot = dot_array[0] + dot_array[1] + dot_array[2] + dot_array[3] +
                dot_array[4] + dot_array[5] + dot_array[6] + dot_array[7];
    float norm_a = norm_a_array[0] + norm_a_array[1] + norm_a_array[2] + norm_a_array[3] +
                   norm_a_array[4] + norm_a_array[5] + norm_a_array[6] + norm_a_array[7];
    float norm_b = norm_b_array[0] + norm_b_array[1] + norm_b_array[2] + norm_b_array[3] +
                   norm_b_array[4] + norm_b_array[5] + norm_b_array[6] + norm_b_array[7];
    for (; i < n; i++) {
        float va = a[i];
        float vb = b[i];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    // If either vector is zero, return π.
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 3.14159265f;
    }

    float cosine = dot / (norm_a * norm_b);

    // Clamp cosine to [-1, 1] first.
    if (cosine > 1.0f) cosine = 1.0f;
    if (cosine < -1.0f) cosine = -1.0f;

    // Additional clamping: if cosine is nearly 1 or -1, force it.
    if (fabsf(1.0f - cosine) < 1e-3f) cosine = 1.0f;
    if (fabsf(cosine + 1.0f) < 1e-3f) cosine = -1.0f;
    return acosf(cosine);
}

/**
 * \brief Computes the dot product distance between two vectors using SIMD instructions.
 *
 * \param a Pointer to the first vector.
 * \param b Pointer to the second vector.
 * \param n The number of elements in the vectors.
 * \return The dot product distance between the two vectors.
 */
float simd_dot_product_distance(const float* a, const float* b, size_t n) {
    size_t i = 0;
    __m256 dot_vec = _mm256_setzero_ps();
    size_t limit = n - (n % 8);
    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#ifdef __FMA__
        dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
#else
        dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va, vb));
#endif
    }
    float dot_array[8];
    _mm256_storeu_ps(dot_array, dot_vec);
    float dot = dot_array[0] + dot_array[1] + dot_array[2] + dot_array[3] +
                dot_array[4] + dot_array[5] + dot_array[6] + dot_array[7];
    for (; i < n; i++) {
        dot += a[i] * b[i];
    }
    // Negate the dot product to get the distance.
    return dot;
}

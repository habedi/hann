#include "simd_distance.h"
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

// Function pointers for distance functions
float (*simd_euclidean_ptr)(const float*, const float*, size_t);
float (*simd_squared_euclidean_ptr)(const float*, const float*, size_t);
float (*simd_manhattan_ptr)(const float*, const float*, size_t);
float (*simd_cosine_distance_ptr)(const float*, const float*, size_t);

// Fallback implementations
float euclidean_fallback(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

float squared_euclidean_fallback(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float manhattan_fallback(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return sum;
}

float cosine_distance_fallback(const float* a, const float* b, size_t n) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < n; i++) {
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

#if defined(__AVX__)
// AVX implementations
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

float euclidean_avx(const float* a, const float* b, size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
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

float squared_euclidean_avx(const float* a, const float* b, size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
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

float manhattan_avx(const float* a, const float* b, size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    size_t i = 0;
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
        sum += fabsf(a[i] - b[i]);
    }
    return sum;
}

float cosine_distance_avx(const float* a, const float* b, size_t n) {
    __m256 dot_vec = _mm256_setzero_ps();
    __m256 norm_a_vec = _mm256_setzero_ps();
    __m256 norm_b_vec = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit = n - (n % 8);
    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va, vb));
        norm_a_vec = _mm256_add_ps(norm_a_vec, _mm256_mul_ps(va, va));
        norm_b_vec = _mm256_add_ps(norm_b_vec, _mm256_mul_ps(vb, vb));
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
#else
float euclidean_avx(const float* a, const float* b, size_t n) { return euclidean_fallback(a, b, n); }
float squared_euclidean_avx(const float* a, const float* b, size_t n) { return squared_euclidean_fallback(a, b, n); }
float manhattan_avx(const float* a, const float* b, size_t n) { return manhattan_fallback(a, b, n); }
float cosine_distance_avx(const float* a, const float* b, size_t n) { return cosine_distance_fallback(a, b, n); }
#endif

#if defined(__AVX2__) && defined(__FMA__)
float euclidean_avx2(const float* a, const float* b, size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit = n - (n % 8);
    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }
    float sum = horizontal_sum256(sum_vec);
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

float squared_euclidean_avx2(const float* a, const float* b, size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit = n - (n % 8);
    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }
    float sum = horizontal_sum256(sum_vec);
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float manhattan_avx2(const float* a, const float* b, size_t n) {
    return manhattan_avx(a, b, n);
}

float cosine_distance_avx2(const float* a, const float* b, size_t n) {
    __m256 dot_vec = _mm256_setzero_ps();
    __m256 norm_a_vec = _mm256_setzero_ps();
    __m256 norm_b_vec = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit = n - (n % 8);
    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
        norm_a_vec = _mm256_fmadd_ps(va, va, norm_a_vec);
        norm_b_vec = _mm256_fmadd_ps(vb, vb, norm_b_vec);
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
#else
float euclidean_avx2(const float* a, const float* b, size_t n) { return euclidean_avx(a, b, n); }
float squared_euclidean_avx2(const float* a, const float* b, size_t n) { return squared_euclidean_avx(a, b, n); }
float manhattan_avx2(const float* a, const float* b, size_t n) { return manhattan_avx(a, b, n); }
float cosine_distance_avx2(const float* a, const float* b, size_t n) { return cosine_distance_avx(a, b, n); }
#endif

void init_distance_functions(int support_level) {
    switch (support_level) {
        case 2: // AVX2
            simd_euclidean_ptr = euclidean_avx2;
            simd_squared_euclidean_ptr = squared_euclidean_avx2;
            simd_manhattan_ptr = manhattan_avx2;
            simd_cosine_distance_ptr = cosine_distance_avx2;
            break;
        case 1: // AVX
            simd_euclidean_ptr = euclidean_avx;
            simd_squared_euclidean_ptr = squared_euclidean_avx;
            simd_manhattan_ptr = manhattan_avx;
            simd_cosine_distance_ptr = cosine_distance_avx;
            break;
        default: // Fallback
            simd_euclidean_ptr = euclidean_fallback;
            simd_squared_euclidean_ptr = squared_euclidean_fallback;
            simd_manhattan_ptr = manhattan_fallback;
            simd_cosine_distance_ptr = cosine_distance_fallback;
            break;
    }
}

// Public functions
float simd_euclidean(const float* a, const float* b, size_t n) {
    return simd_euclidean_ptr(a, b, n);
}

float simd_squared_euclidean(const float* a, const float* b, size_t n) {
    return simd_squared_euclidean_ptr(a, b, n);
}

float simd_manhattan(const float* a, const float* b, size_t n) {
    return simd_manhattan_ptr(a, b, n);
}

float simd_cosine_distance(const float* a, const float* b, size_t n) {
    return simd_cosine_distance_ptr(a, b, n);
}

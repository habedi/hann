#include "simd_distance.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>

// The AVX variants are compiled with a per-function target attribute instead
// of a package-wide -mavx flag, so the fallback functions and the dispatch
// code never contain AVX instructions and stay safe on CPUs without AVX.
// hann_cpu_init still selects a variant at runtime through function pointers.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define HANN_TARGET_AVX __attribute__((target("avx")))
#define HANN_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif

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

#ifdef HANN_TARGET_AVX
// AVX implementations
HANN_TARGET_AVX
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

HANN_TARGET_AVX
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

HANN_TARGET_AVX
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

HANN_TARGET_AVX
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

HANN_TARGET_AVX
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

#ifdef HANN_TARGET_AVX2
// AVX2 implementations. The multiply-add accumulations use FMA, so the
// target attribute enables both avx2 and fma, and hann_cpu_init selects
// this tier only when the CPU reports both features. Each accumulation
// runs over four independent accumulators, because a single accumulator
// chains every fmadd through the previous one and the loop then runs at
// FMA latency instead of FMA throughput.
HANN_TARGET_AVX2
float euclidean_avx2(const float* a, const float* b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit32 = n - (n % 32);
    for (; i < limit32; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    size_t limit = n - (n % 8);
    for (; i < limit; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }
    __m256 sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float sum = horizontal_sum256(sum_vec);
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

HANN_TARGET_AVX2
float squared_euclidean_avx2(const float* a, const float* b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit32 = n - (n % 32);
    for (; i < limit32; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    size_t limit = n - (n % 8);
    for (; i < limit; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }
    __m256 sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float sum = horizontal_sum256(sum_vec);
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// The Manhattan kernel accumulates absolute differences, which has no
// multiply-add shape for FMA to improve, so the AVX2 tier reuses the AVX
// variant.
float manhattan_avx2(const float* a, const float* b, size_t n) {
    return manhattan_avx(a, b, n);
}

HANN_TARGET_AVX2
float cosine_distance_avx2(const float* a, const float* b, size_t n) {
    __m256 dot0 = _mm256_setzero_ps(), dot1 = _mm256_setzero_ps();
    __m256 dot2 = _mm256_setzero_ps(), dot3 = _mm256_setzero_ps();
    __m256 na0 = _mm256_setzero_ps(), na1 = _mm256_setzero_ps();
    __m256 na2 = _mm256_setzero_ps(), na3 = _mm256_setzero_ps();
    __m256 nb0 = _mm256_setzero_ps(), nb1 = _mm256_setzero_ps();
    __m256 nb2 = _mm256_setzero_ps(), nb3 = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit32 = n - (n % 32);
    for (; i < limit32; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        dot0 = _mm256_fmadd_ps(va0, vb0, dot0);
        na0 = _mm256_fmadd_ps(va0, va0, na0);
        nb0 = _mm256_fmadd_ps(vb0, vb0, nb0);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        dot1 = _mm256_fmadd_ps(va1, vb1, dot1);
        na1 = _mm256_fmadd_ps(va1, va1, na1);
        nb1 = _mm256_fmadd_ps(vb1, vb1, nb1);
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        dot2 = _mm256_fmadd_ps(va2, vb2, dot2);
        na2 = _mm256_fmadd_ps(va2, va2, na2);
        nb2 = _mm256_fmadd_ps(vb2, vb2, nb2);
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        dot3 = _mm256_fmadd_ps(va3, vb3, dot3);
        na3 = _mm256_fmadd_ps(va3, va3, na3);
        nb3 = _mm256_fmadd_ps(vb3, vb3, nb3);
    }
    size_t limit = n - (n % 8);
    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot0 = _mm256_fmadd_ps(va, vb, dot0);
        na0 = _mm256_fmadd_ps(va, va, na0);
        nb0 = _mm256_fmadd_ps(vb, vb, nb0);
    }
    __m256 dot_vec = _mm256_add_ps(_mm256_add_ps(dot0, dot1), _mm256_add_ps(dot2, dot3));
    __m256 norm_a_vec = _mm256_add_ps(_mm256_add_ps(na0, na1), _mm256_add_ps(na2, na3));
    __m256 norm_b_vec = _mm256_add_ps(_mm256_add_ps(nb0, nb1), _mm256_add_ps(nb2, nb3));
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

#include "simd_ops.h"
#include "simd_distance.h"
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

// Function pointer for the normalization function
void (*simd_normalize_ptr)(float*, size_t);

// Fallback implementation for normalization
void normalize_fallback(float *vec, size_t len) {
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum += vec[i] * vec[i];
    }
    float norm = sqrtf(sum);
    if (norm == 0.0f) return;
    for (size_t i = 0; i < len; i++) {
        vec[i] /= norm;
    }
}

// AVX implementation for normalization
#ifdef __AVX__
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

void normalize_avx(float *vec, size_t len) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit = len - (len % 8);
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v, v));
    }
    float total = horizontal_sum256(sum);
    for (; i < len; i++) {
        total += vec[i] * vec[i];
    }
    float norm = sqrtf(total);
    if (norm == 0.0f) return;
    __m256 norm_vec = _mm256_set1_ps(norm);
    i = 0;
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        v = _mm256_div_ps(v, norm_vec);
        _mm256_storeu_ps(&vec[i], v);
    }
    for (; i < len; i++) {
        vec[i] /= norm;
    }
}
#else
void normalize_avx(float *vec, size_t len) {
    normalize_fallback(vec, len);
}
#endif

// AVX2 implementation for normalization
#if defined(__AVX2__) && defined(__FMA__)
void normalize_avx2(float *vec, size_t len) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    size_t limit = len - (len % 8);
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        sum = _mm256_fmadd_ps(v, v, sum);
    }
    float total = horizontal_sum256(sum);
    for (; i < len; i++) {
        total += vec[i] * vec[i];
    }
    float norm = sqrtf(total);
    if (norm == 0.0f) return;
    __m256 norm_vec = _mm256_set1_ps(norm);
    i = 0;
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        v = _mm256_div_ps(v, norm_vec);
        _mm256_storeu_ps(&vec[i], v);
    }
    for (; i < len; i++) {
        vec[i] /= norm;
    }
}
#else
void normalize_avx2(float *vec, size_t len) {
    normalize_avx(vec, len);
}
#endif

void hann_cpu_init(int support_level) {
    switch (support_level) {
        case 2: // AVX2
            simd_normalize_ptr = normalize_avx2;
            break;
        case 1: // AVX
            simd_normalize_ptr = normalize_avx;
            break;
        default: // Fallback
            simd_normalize_ptr = normalize_fallback;
            break;
    }
    init_distance_functions(support_level);
}

void simd_normalize(float *vec, size_t len) {
    simd_normalize_ptr(vec, len);
}

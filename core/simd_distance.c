#include "simd_distance.h"
#include "simd_isa.h"
#include <math.h>
#include <stddef.h>

// The AVX variants are compiled with a per-function target attribute
// (declared in simd_isa.h) instead of a package-wide -mavx flag, so the
// fallback functions and the dispatch code never contain AVX instructions
// and stay safe on CPUs without AVX. hann_cpu_init still selects a variant
// at runtime through function pointers. The vector bodies live in
// simd_kernels.inc.h and are instantiated per ISA below.

// Function pointers for distance functions
float (*simd_euclidean_ptr)(const float*, const float*, size_t);
float (*simd_squared_euclidean_ptr)(const float*, const float*, size_t);
float (*simd_manhattan_ptr)(const float*, const float*, size_t);
float (*simd_cosine_distance_ptr)(const float*, const float*, size_t);

// Fallback implementations. They stay hand-written scalar C, as the
// readable reference for the vector kernels.
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

#define HANN_EMIT_DISTANCE 1

#ifdef HANN_TARGET_AVX
// AVX instantiations
#define HANN_EMIT_MANHATTAN 1
#include "simd_kernels_avx.inc.h"
#undef HANN_EMIT_MANHATTAN
#else
float euclidean_avx(const float* a, const float* b, size_t n) { return euclidean_fallback(a, b, n); }
float squared_euclidean_avx(const float* a, const float* b, size_t n) { return squared_euclidean_fallback(a, b, n); }
float manhattan_avx(const float* a, const float* b, size_t n) { return manhattan_fallback(a, b, n); }
float cosine_distance_avx(const float* a, const float* b, size_t n) { return cosine_distance_fallback(a, b, n); }
#endif

#ifdef HANN_TARGET_AVX2
// AVX2 instantiations
#include "simd_kernels_avx2.inc.h"
#else
float euclidean_avx2(const float* a, const float* b, size_t n) { return euclidean_avx(a, b, n); }
float squared_euclidean_avx2(const float* a, const float* b, size_t n) { return squared_euclidean_avx(a, b, n); }
float cosine_distance_avx2(const float* a, const float* b, size_t n) { return cosine_distance_avx(a, b, n); }
#endif

#undef HANN_EMIT_DISTANCE

// The Manhattan kernel accumulates absolute differences, which has no
// multiply-add shape for FMA to improve, so the AVX2 tier reuses the AVX
// variant. Aliasing is a dispatch decision, so it lives here rather than in
// the kernel bodies.
float manhattan_avx2(const float* a, const float* b, size_t n) {
    return manhattan_avx(a, b, n);
}

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

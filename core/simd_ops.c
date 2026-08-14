#include "simd_ops.h"
#include "simd_distance.h"
#include "simd_isa.h"
#include <math.h>
#include <stddef.h>

// The AVX variants are compiled with a per-function target attribute
// (declared in simd_isa.h) instead of a package-wide -mavx flag, so the
// fallback function and hann_cpu_init never contain AVX instructions and
// stay safe on CPUs without AVX. The vector body lives in
// simd_kernels.inc.h and is instantiated per ISA below.

// Function pointer for the normalization function
void (*simd_normalize_ptr)(float*, size_t);

// Fallback implementation for normalization. It stays hand-written scalar
// C, as the readable reference for the vector kernel.
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

#define HANN_EMIT_NORMALIZE 1

#ifdef HANN_TARGET_AVX
// AVX instantiation
#include "simd_kernels_avx.inc.h"
#else
void normalize_avx(float *vec, size_t len) {
    normalize_fallback(vec, len);
}
#endif

#ifdef HANN_TARGET_AVX2
// AVX2 instantiation
#include "simd_kernels_avx2.inc.h"
#else
void normalize_avx2(float *vec, size_t len) {
    normalize_avx(vec, len);
}
#endif

#undef HANN_EMIT_NORMALIZE

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

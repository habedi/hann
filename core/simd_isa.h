// simd_isa.h declares the per-function target attribute macros and the helpers shared by every SIMD kernel
// instantiation. The AVX variants are compiled with a per-function target attribute instead of a package-wide
// -mavx flag, so the fallback functions and the dispatch code never contain AVX instructions and stay safe on
// CPUs without AVX.
#ifndef SIMD_ISA_H
#define SIMD_ISA_H

#include <stddef.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define HANN_TARGET_AVX __attribute__((target("avx")))
#define HANN_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif

// HANN_FN expands a kernel base name into its per-ISA name by appending the HANN_SUFFIX defined by the current
// vocabulary, so HANN_FN(euclidean) becomes euclidean_avx in the AVX instantiation.
#define HANN_CAT_(a, b) a##_##b
#define HANN_CAT(a, b) HANN_CAT_(a, b)
#define HANN_FN(name) HANN_CAT(name, HANN_SUFFIX)

#ifdef HANN_TARGET_AVX
// horizontal_sum256 reduces the eight lanes of a 256-bit vector to one float. It carries the AVX target attribute,
// which is a subset of the AVX2 target, so it inlines into both instantiation tiers.
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
#endif

#endif // SIMD_ISA_H

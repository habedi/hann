#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

// Normalizes a single vector using AVX instructions
void avx_normalize(float *vec, int len);

#ifdef __cplusplus
}
#endif

#endif // SIMD_OPS_H

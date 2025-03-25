#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Normalizes a single vector using AVX.
void avx_normalize(float *vec, size_t len);

#ifdef __cplusplus
}
#endif

#endif // SIMD_OPS_H

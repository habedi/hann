#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the SIMD operations based on CPU support.
void hann_cpu_init(int support_level);

// Normalizes a single vector.
void simd_normalize(float *vec, size_t len);

#ifdef __cplusplus
}
#endif

#endif // SIMD_OPS_H

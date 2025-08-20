#ifndef SIMD_DISTANCE_H
#define SIMD_DISTANCE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Function pointers for distance functions
extern float (*simd_euclidean_ptr)(const float*, const float*, size_t);
extern float (*simd_squared_euclidean_ptr)(const float*, const float*, size_t);
extern float (*simd_manhattan_ptr)(const float*, const float*, size_t);
extern float (*simd_cosine_distance_ptr)(const float*, const float*, size_t);

// Public functions that call the function pointers
float simd_euclidean(const float* a, const float* b, size_t n);
float simd_squared_euclidean(const float* a, const float* b, size_t n);
float simd_manhattan(const float* a, const float* b, size_t n);
float simd_cosine_distance(const float* a, const float* b, size_t n);

void init_distance_functions(int support_level);

#ifdef __cplusplus
}
#endif

#endif // SIMD_DISTANCE_H

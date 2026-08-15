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

// Function pointers for the batch variants, which compute the distance from
// one query to n candidate vectors stored consecutively in a flat buffer of
// n rows of dim floats, writing n doubles to out.
extern void (*simd_euclidean_batch_ptr)(const float*, const float*, size_t, size_t, double*);
extern void (*simd_squared_euclidean_batch_ptr)(const float*, const float*, size_t, size_t, double*);
extern void (*simd_manhattan_batch_ptr)(const float*, const float*, size_t, size_t, double*);
extern void (*simd_cosine_distance_batch_ptr)(const float*, const float*, size_t, size_t, double*);

// Public functions that call the function pointers
float simd_euclidean(const float* a, const float* b, size_t n);
float simd_squared_euclidean(const float* a, const float* b, size_t n);
float simd_manhattan(const float* a, const float* b, size_t n);
float simd_cosine_distance(const float* a, const float* b, size_t n);

void simd_euclidean_batch(const float* q, const float* flat, size_t dim, size_t n, double* out);
void simd_squared_euclidean_batch(const float* q, const float* flat, size_t dim, size_t n, double* out);
void simd_manhattan_batch(const float* q, const float* flat, size_t dim, size_t n, double* out);
void simd_cosine_distance_batch(const float* q, const float* flat, size_t dim, size_t n, double* out);

void init_distance_functions(int support_level);

#ifdef __cplusplus
}
#endif

#endif // SIMD_DISTANCE_H

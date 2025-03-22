#ifndef SIMD_DISTANCE_H
#define SIMD_DISTANCE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Computes the Euclidean distance between two float arrays using AVX.
float simd_euclidean(const float* a, const float* b, size_t n);

// Computes the squared Euclidean distance (without the square root) between two float arrays using AVX.
float simd_squared_euclidean(const float* a, const float* b, size_t n);

// Computes the Manhattan (L1) distance between two float arrays using AVX.
float simd_manhattan(const float* a, const float* b, size_t n);

// Computes the cosine distance between two float arrays using AVX.
float simd_cosine_distance(const float* a, const float* b, size_t n);

// Computes the angular distance (in radians) between two float arrays using AVX.
float simd_angular_distance(const float* a, const float* b, size_t n);

// Computes the dot product distance between two float arrays using AVX.
float simd_dot_product_distance(const float* a, const float* b, size_t n);

#ifdef __cplusplus
}
#endif

#endif // SIMD_DISTANCE_H

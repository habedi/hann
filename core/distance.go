package core

/*
#cgo CFLAGS: -O2
#cgo LDFLAGS: -lm
#include "simd_distance.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

// DistanceFunc computes the distance between two vectors.
// a: the first vector.
// b: the second vector.
// Returns the computed distance as a float64 and an error if validation fails.
type DistanceFunc func(a, b []float32) (float64, error)

// euclidean computes the Euclidean (L2) distance between two vectors.
func euclidean(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("vectors must have the same length")
	}
	if len(a) == 0 {
		return 0, nil
	}
	n := C.size_t(len(a))
	dist := float64(C.simd_euclidean(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
	return dist, nil
}

// squaredEuclidean computes the squared Euclidean distance between two vectors.
func squaredEuclidean(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("vectors must have the same length")
	}
	if len(a) == 0 {
		return 0, nil
	}
	n := C.size_t(len(a))
	dist := float64(C.simd_squared_euclidean(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
	return dist, nil
}

// manhattan computes the Manhattan (L1) distance between two vectors.
func manhattan(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("vectors must have the same length")
	}
	if len(a) == 0 {
		return 0, nil
	}
	n := C.size_t(len(a))
	dist := float64(C.simd_manhattan(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
	return dist, nil
}

// cosineDistance computes the cosine distance between two vectors.
func cosineDistance(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("vectors must have the same length")
	}
	if len(a) == 0 {
		return 0, nil
	}
	n := C.size_t(len(a))
	dist := float64(C.simd_cosine_distance(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
	return dist, nil
}

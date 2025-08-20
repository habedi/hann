package core

/*
#cgo CFLAGS: -O2 -mavx
#cgo LDFLAGS: -lm
#include "simd_distance.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Distances is a map of human–readable names to distance functions.
// You can use it to choose a distance metric by name.
var Distances = map[string]DistanceFunc{
	"euclidean":         Euclidean,
	"squared_euclidean": SquaredEuclidean,
	"manhattan":         Manhattan,
	"cosine":            CosineDistance,
}

// DistanceFunc computes the distance between two vectors.
// a: the first vector.
// b: the second vector.
// Returns the computed distance as a float64 and an error if validation fails.
type DistanceFunc func(a, b []float32) (float64, error)

// Euclidean computes the Euclidean (L2) distance between two vectors.
func Euclidean(a, b []float32) (float64, error) {
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

// SquaredEuclidean computes the squared Euclidean distance between two vectors.
func SquaredEuclidean(a, b []float32) (float64, error) {
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

// Manhattan computes the Manhattan (L1) distance between two vectors.
func Manhattan(a, b []float32) (float64, error) {
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

// CosineDistance computes the cosine distance between two vectors.
func CosineDistance(a, b []float32) (float64, error) {
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

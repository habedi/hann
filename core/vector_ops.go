package core

/*
#cgo CFLAGS: -O2 -mavx
#cgo LDFLAGS: -lm
#include "simd_ops.h"
*/
import "C"
import "unsafe"

// NormalizeVector normalizes a single float32 slice using AVX instructions.
func NormalizeVector(vec []float32) {
	if len(vec) == 0 {
		return
	}
	C.avx_normalize((*C.float)(unsafe.Pointer(&vec[0])), C.int(len(vec)))
}

// NormalizeBatch normalizes multiple vectors in a batch using goroutines.
func NormalizeBatch(vecs [][]float32) {
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return
	}

	// Create a channel to synchronize the goroutines.
	done := make(chan struct{})
	for i := range vecs {
		go func(i int) {
			C.avx_normalize((*C.float)(unsafe.Pointer(&vecs[i][0])), C.int(len(vecs[i])))
			done <- struct{}{}
		}(i)
	}

	// Wait for all go routines to finish.
	for range vecs {
		<-done
	}

	close(done)
}

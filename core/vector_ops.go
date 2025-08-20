package core

/*
#cgo CFLAGS: -O2 -mavx
#cgo LDFLAGS: -lm
#include "simd_ops.h"
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// NormalizeVector normalizes a single float32 slice using AVX instructions.
func NormalizeVector(vec []float32) {
	if len(vec) == 0 {
		return
	}
	C.simd_normalize((*C.float)(unsafe.Pointer(&vec[0])), C.size_t(len(vec)))
}

// NormalizeBatch normalizes multiple vectors in parallel using a worker pool.
func NormalizeBatch(vecs [][]float32) {
	if len(vecs) == 0 {
		return
	}

	numWorkers := runtime.NumCPU()
	tasks := make(chan int, len(vecs))
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range tasks {
				if len(vecs[idx]) > 0 {
					C.simd_normalize((*C.float)(unsafe.Pointer(&vecs[idx][0])), C.size_t(len(vecs[idx])))
				}
			}
		}()
	}

	// Feed tasks
	for i := range vecs {
		tasks <- i
	}
	close(tasks)

	// Wait for all workers to finish
	wg.Wait()
}

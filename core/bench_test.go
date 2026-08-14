package core

import (
	"fmt"
	"math/rand"
	"testing"
)

// benchDims covers the dimensions of the example datasets, from the smallest
// to MNIST-sized vectors.
var benchDims = []int{25, 128, 200, 784}

// benchVector returns a vector of the given dimension filled with values from
// the given generator.
func benchVector(rng *rand.Rand, dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}
	return vec
}

// BenchmarkDistance measures every built-in metric's Distance at each
// benchmark dimension, so the fallback, AVX, and AVX2 kernels can be compared
// across machines.
func BenchmarkDistance(b *testing.B) {
	metrics := []Metric{Euclidean, SquaredEuclidean, Manhattan, Cosine}
	for _, metric := range metrics {
		for _, dim := range benchDims {
			b.Run(fmt.Sprintf("%s/%d", metric.Name(), dim), func(b *testing.B) {
				rng := rand.New(rand.NewSource(42))
				va := benchVector(rng, dim)
				vb := benchVector(rng, dim)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := metric.Distance(va, vb); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkNormalizeVector measures NormalizeVector at each benchmark
// dimension.
func BenchmarkNormalizeVector(b *testing.B) {
	for _, dim := range benchDims {
		b.Run(fmt.Sprintf("%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			src := benchVector(rng, dim)
			buf := make([]float32, dim)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// NormalizeVector mutates its argument, so each iteration
				// copies the source into a reused buffer first. The copy is
				// included in the measured time, so the reported figure is a
				// slight overestimate of normalization alone.
				copy(buf, src)
				NormalizeVector(buf)
			}
		})
	}
}

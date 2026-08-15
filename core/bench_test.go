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
// benchmark dimension. This lets the fallback, AVX, and AVX2 kernels be
// compared across machines.
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

// benchBatchSize and benchBatchDim size the batch benchmarks: one query
// against a thousand candidates at the dimension of the SIFT-like datasets.
const (
	benchBatchSize = 1000
	benchBatchDim  = 128
)

// benchBatchMetrics are the metrics the batch benchmarks compare.
var benchBatchMetrics = []Metric{Euclidean, Cosine}

// BenchmarkDistanceBatch measures one DistanceBatch call over a thousand
// candidate vectors. BenchmarkDistanceSingle makes the same thousand
// comparisons through per-pair Distance calls. Comparing the two shows the
// gain from batching alone. The flat buffer is packed in setup, so neither
// benchmark counts the packing cost. The rpt search benchmark counts it in
// the consumer.
func BenchmarkDistanceBatch(b *testing.B) {
	for _, metric := range benchBatchMetrics {
		b.Run(metric.Name(), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			query := benchVector(rng, benchBatchDim)
			flat := benchVector(rng, benchBatchSize*benchBatchDim)
			out := make([]float64, benchBatchSize)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := metric.DistanceBatch(query, flat, out); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkDistanceSingle measures a thousand per-pair Distance calls over
// the same data as BenchmarkDistanceBatch. One iteration of either benchmark
// makes the same comparisons.
func BenchmarkDistanceSingle(b *testing.B) {
	for _, metric := range benchBatchMetrics {
		b.Run(metric.Name(), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			query := benchVector(rng, benchBatchDim)
			flat := benchVector(rng, benchBatchSize*benchBatchDim)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := 0; j < benchBatchSize; j++ {
					row := flat[j*benchBatchDim : (j+1)*benchBatchDim]
					if _, err := metric.Distance(query, row); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
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
				// NormalizeVector changes its argument, so each iteration
				// copies the source into a reused buffer first. The copy is
				// included in the measured time, so the reported time is
				// slightly more than normalization alone.
				copy(buf, src)
				NormalizeVector(buf)
			}
		})
	}
}

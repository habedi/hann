package core

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// batchTestCounts and batchTestDims span single-candidate, small, and
// SIMD-wide batches over the dimensions of the example datasets.
var (
	batchTestCounts = []int{1, 7, 64}
	batchTestDims   = []int{25, 128, 784}
)

// randomBatch returns a query and a flat buffer of n candidate vectors of the
// given dimension, filled from the given generator.
func randomBatch(rng *rand.Rand, dim, n int) (query, flat []float32) {
	query = make([]float32, dim)
	for i := range query {
		query[i] = rng.Float32()*20 - 10
	}
	flat = make([]float32, n*dim)
	for i := range flat {
		flat[i] = rng.Float32()*20 - 10
	}
	return query, flat
}

// TestDistanceBatchParity checks that DistanceBatch and RankBatch agree with
// the per-pair Distance and Rank for every built-in metric over randomized
// batches.
func TestDistanceBatchParity(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, m := range []Metric{Euclidean, SquaredEuclidean, Manhattan, Cosine} {
		for _, dim := range batchTestDims {
			for _, n := range batchTestCounts {
				query, flat := randomBatch(rng, dim, n)
				distOut := make([]float64, n)
				if err := m.DistanceBatch(query, flat, distOut); err != nil {
					t.Fatalf("%s dim %d n %d: DistanceBatch: %v", m.Name(), dim, n, err)
				}
				rankOut := make([]float64, n)
				if err := m.RankBatch(query, flat, rankOut); err != nil {
					t.Fatalf("%s dim %d n %d: RankBatch: %v", m.Name(), dim, n, err)
				}
				for i := 0; i < n; i++ {
					row := flat[i*dim : (i+1)*dim]
					wantDist, err := m.Distance(query, row)
					if err != nil {
						t.Fatalf("%s dim %d n %d: Distance: %v", m.Name(), dim, n, err)
					}
					tol := 1e-6 * math.Max(1, math.Abs(wantDist))
					if !almostEqual(distOut[i], wantDist, tol) {
						t.Errorf("%s dim %d n %d row %d: DistanceBatch %v, Distance %v",
							m.Name(), dim, n, i, distOut[i], wantDist)
					}
					wantRank, err := m.Rank(query, row)
					if err != nil {
						t.Fatalf("%s dim %d n %d: Rank: %v", m.Name(), dim, n, err)
					}
					tol = 1e-6 * math.Max(1, math.Abs(wantRank))
					if !almostEqual(rankOut[i], wantRank, tol) {
						t.Errorf("%s dim %d n %d row %d: RankBatch %v, Rank %v",
							m.Name(), dim, n, i, rankOut[i], wantRank)
					}
				}
			}
		}
	}
}

// TestDistanceBatchErrors covers the validation of the batch arguments and
// the empty-batch no-op for both batch methods.
func TestDistanceBatchErrors(t *testing.T) {
	methods := []struct {
		name string
		call func(query, flat []float32, out []float64) error
	}{
		{"DistanceBatch", Euclidean.DistanceBatch},
		{"RankBatch", Euclidean.RankBatch},
	}
	for _, method := range methods {
		t.Run(method.name, func(t *testing.T) {
			if err := method.call(nil, []float32{1, 2}, make([]float64, 1)); err == nil {
				t.Errorf("empty query: expected an error but got nil")
			}
			if err := method.call([]float32{1, 2}, []float32{1, 2, 3}, make([]float64, 1)); err == nil {
				t.Errorf("ragged flat: expected an error but got nil")
			}
			if err := method.call([]float32{1, 2}, []float32{1, 2, 3, 4}, make([]float64, 3)); err == nil {
				t.Errorf("wrong out length: expected an error but got nil")
			}
			if err := method.call([]float32{1, 2}, nil, nil); err != nil {
				t.Errorf("empty batch: expected a no-op but got %v", err)
			}
		})
	}
}

// TestDistanceBatchCustomMetric checks that a metric created with NewMetric,
// which has no batch kernel, computes batches through the per-pair fallback
// loop for both batch methods.
func TestDistanceBatchCustomMetric(t *testing.T) {
	dot := func(a, b []float32) (float64, error) {
		sum := 0.0
		for i := range a {
			sum += float64(a[i]) * float64(b[i])
		}
		return -sum, nil
	}
	m := NewMetric("negative_dot_batch_test", dot, false)
	rng := rand.New(rand.NewSource(13))
	const dim, n = 25, 7
	query, flat := randomBatch(rng, dim, n)
	distOut := make([]float64, n)
	if err := m.DistanceBatch(query, flat, distOut); err != nil {
		t.Fatalf("DistanceBatch: %v", err)
	}
	rankOut := make([]float64, n)
	if err := m.RankBatch(query, flat, rankOut); err != nil {
		t.Fatalf("RankBatch: %v", err)
	}
	for i := 0; i < n; i++ {
		want, err := dot(query, flat[i*dim:(i+1)*dim])
		if err != nil {
			t.Fatal(err)
		}
		if !almostEqual(distOut[i], want, 1e-9) {
			t.Errorf("row %d: DistanceBatch %v, per-pair %v", i, distOut[i], want)
		}
		if !almostEqual(rankOut[i], want, 1e-9) {
			t.Errorf("row %d: RankBatch %v, per-pair %v", i, rankOut[i], want)
		}
	}
}

// TestDistanceBatchLoopError checks that the per-pair fallback loop stops at
// the first candidate whose distance computation fails and returns that error
// to the caller, for both batch methods.
func TestDistanceBatchLoopError(t *testing.T) {
	failure := errors.New("distance failure injected by the test")
	// The function fails on any candidate whose first element is negative, so
	// the loop succeeds on the first row and fails on the second.
	fn := func(a, b []float32) (float64, error) {
		if b[0] < 0 {
			return 0, failure
		}
		return float64(a[0]) + float64(b[0]), nil
	}
	m := NewMetric("failing_batch_test", fn, false)
	query := []float32{1, 1}
	flat := []float32{2, 2, -3, 3, 4, 4}
	out := make([]float64, 3)
	if err := m.DistanceBatch(query, flat, out); !errors.Is(err, failure) {
		t.Errorf("DistanceBatch: expected the injected error, got %v", err)
	}
	if out[0] != 3 {
		t.Errorf("DistanceBatch: expected the first row to be computed before the failure, got %v", out[0])
	}
	if err := m.RankBatch(query, flat, out); !errors.Is(err, failure) {
		t.Errorf("RankBatch: expected the injected error, got %v", err)
	}
}

// Package testutil provides shared helpers for testing the index packages:
// deterministic synthetic data, brute-force ground truth, recall computation,
// a property-based operation runner, and a concurrency stress runner.
package testutil

import (
	"math"
	"math/rand"
	"sort"

	"github.com/habedi/hann/core"
)

// ClusteredData returns n vectors of the given dimension drawn from the given
// number of Gaussian clusters, keyed by ids 0 through n-1. The same seed
// returns the same data. Clustered data is used instead of uniform data
// because uniform random points in high dimensions have nearly equidistant
// neighbors, which makes recall measurements meaningless.
func ClusteredData(seed int64, n, dim, clusters int) map[int][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range centers {
		center := make([]float32, dim)
		for d := range center {
			center[d] = rng.Float32()*20 - 10
		}
		centers[c] = center
	}
	data := make(map[int][]float32, n)
	for id := 0; id < n; id++ {
		center := centers[rng.Intn(clusters)]
		vec := make([]float32, dim)
		for d := range vec {
			vec[d] = center[d] + float32(rng.NormFloat64())
		}
		data[id] = vec
	}
	return data
}

// Queries returns q query vectors placed near randomly chosen data points, so
// each query has meaningful nearest neighbors. The same seed returns the same
// queries.
func Queries(seed int64, data map[int][]float32, q int) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	ids := make([]int, 0, len(data))
	for id := range data {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	queries := make([][]float32, q)
	for i := range queries {
		base := data[ids[rng.Intn(len(ids))]]
		vec := make([]float32, len(base))
		for d := range vec {
			vec[d] = base[d] + float32(rng.NormFloat64())*0.1
		}
		queries[i] = vec
	}
	return queries
}

// BruteForceKNN returns the ids of the k nearest data points to the query
// under the given distance, breaking distance ties by id, so the result is
// deterministic.
func BruteForceKNN(query []float32, data map[int][]float32, k int, dist core.DistanceFunc) ([]int, error) {
	type scored struct {
		id   int
		dist float64
	}
	all := make([]scored, 0, len(data))
	for id, vec := range data {
		d, err := dist(query, vec)
		if err != nil {
			return nil, err
		}
		all = append(all, scored{id: id, dist: d})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].dist == all[j].dist {
			return all[i].id < all[j].id
		}
		return all[i].dist < all[j].dist
	})
	if k > len(all) {
		k = len(all)
	}
	ids := make([]int, k)
	for i := 0; i < k; i++ {
		ids[i] = all[i].id
	}
	return ids, nil
}

// Recall returns the fraction of want ids that appear in got.
func Recall(got []core.Neighbor, want []int) float64 {
	if len(want) == 0 {
		return 0
	}
	gotSet := make(map[int]struct{}, len(got))
	for _, n := range got {
		gotSet[n.ID] = struct{}{}
	}
	hits := 0
	for _, id := range want {
		if _, ok := gotSet[id]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(want))
}

// CopyVector returns a copy of the vector, so index operations that store or
// normalize their argument in place cannot alias test-owned data.
func CopyVector(vec []float32) []float32 {
	out := make([]float32, len(vec))
	copy(out, vec)
	return out
}

// almostEqual reports whether two distances agree within a relative tolerance,
// which is needed because the SIMD variants and the fallback do not produce
// bit-identical results.
func almostEqual(a, b float64) bool {
	diff := math.Abs(a - b)
	if diff < 1e-4 {
		return true
	}
	scale := math.Max(math.Abs(a), math.Abs(b))
	return diff <= 1e-3*scale
}

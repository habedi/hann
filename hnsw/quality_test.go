package hnsw_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/hnsw"
	"github.com/habedi/hann/internal/testutil"
)

// measureMeanRecall builds indexes over clustered data with the given
// metric. It returns the mean recall at k=10 over 20 queries per build
// against the brute-force ground truth. The graph shape depends on the
// package-level level generator, so recall varies between builds.
// Averaging over three builds narrows the spread the assertion has to
// allow for.
func measureMeanRecall(t *testing.T, metric core.Metric) float64 {
	t.Helper()
	const (
		dim      = 16
		n        = 2000
		clusters = 16
		q        = 20
		k        = 10
		builds   = 3
	)
	total := 0.0
	for b := 0; b < builds; b++ {
		dataSeed := int64(42 + 100*b)
		data := testutil.ClusteredData(dataSeed, n, dim, clusters)
		index := newTestIndex(t, dim, hnsw.WithM(16), hnsw.WithEf(100), hnsw.WithMetric(metric))
		// The index normalizes cosine vectors in place. It therefore gets
		// copies, and the ground truth keeps the raw data.
		vectors := make(map[int][]float32, len(data))
		for id, vec := range data {
			vectors[id] = testutil.CopyVector(vec)
		}
		if err := index.BulkAdd(vectors); err != nil {
			t.Fatalf("BulkAdd failed: %v", err)
		}
		queries := testutil.Queries(dataSeed+1, data, q)
		for _, query := range queries {
			want, err := testutil.BruteForceKNN(query, data, k, metric)
			if err != nil {
				t.Fatalf("BruteForceKNN failed: %v", err)
			}
			got, err := index.Search(testutil.CopyVector(query), k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}
			total += testutil.Recall(got, want)
		}
	}
	return total / float64(q*builds)
}

func TestHNSWIndex_RecallEuclidean(t *testing.T) {
	recall := measureMeanRecall(t, core.Euclidean)
	t.Logf("euclidean mean recall at k=10: %.4f", recall)
	// Observed value over 12 runs: 1.00 on every run. The neighbor selection
	// heuristic raised the range from the 0.64 to 0.93 seen with closest-M
	// selection. It does so by keeping the level 0 graph connected across
	// clusters. The threshold is a regression tripwire, not a quality goal.
	if recall < 0.90 {
		t.Errorf("euclidean mean recall %.4f is below the regression threshold 0.90", recall)
	}
}

func TestHNSWIndex_RecallCosine(t *testing.T) {
	recall := measureMeanRecall(t, core.Cosine)
	t.Logf("cosine mean recall at k=10: %.4f", recall)
	// Observed value over 12 runs: 1.00 on every run. The neighbor selection
	// heuristic raised the range from the 0.55 to 0.92 seen with closest-M
	// selection. The threshold is a regression tripwire, not a quality goal.
	if recall < 0.90 {
		t.Errorf("cosine mean recall %.4f is below the regression threshold 0.90", recall)
	}
}

// TestHNSWIndex_Level0Connectivity checks that the level 0 graph stays
// connected on clustered data. A search whose depth covers the whole index
// must reach every node through the graph alone, so the brute-force
// fallback counter has to stay at zero. Closest-M neighbor selection used
// to split the graph into per-cluster components. This test would catch
// that. Graph construction is unseeded and HNSW connectivity is
// probabilistic, so a rare build can strand a node. The test therefore
// builds up to three independent indexes and requires at least one fully
// connected graph. The fragmentation regression can never produce one.
func TestHNSWIndex_Level0Connectivity(t *testing.T) {
	const (
		dim      = 16
		n        = 2000
		clusters = 16
		attempts = 3
	)
	data := testutil.ClusteredData(42, n, dim, clusters)
	query := testutil.Queries(43, data, 1)[0]
	for attempt := 1; attempt <= attempts; attempt++ {
		index := newTestIndex(t, dim, hnsw.WithM(16), hnsw.WithEf(n))
		vectors := make(map[int][]float32, len(data))
		for id, vec := range data {
			vectors[id] = testutil.CopyVector(vec)
		}
		if err := index.BulkAdd(vectors); err != nil {
			t.Fatalf("attempt %d: BulkAdd failed: %v", attempt, err)
		}
		results, err := index.Search(testutil.CopyVector(query), n)
		if err != nil {
			t.Fatalf("attempt %d: Search failed: %v", attempt, err)
		}
		if len(results) != n {
			t.Fatalf("attempt %d: search with k=%d returned %d results", attempt, n, len(results))
		}
		fb := index.Stats().FallbackSearches
		if fb == 0 {
			return
		}
		t.Logf("attempt %d: graph search reached only part of the index (%d fallback searches)", attempt, fb)
	}
	t.Errorf("no fully connected graph in %d builds; the level 0 graph is fragmenting", attempts)
}

// TestHNSWIndex_DifferentialExact compares complete searches against brute
// force for both supported distances. With k equal to the index size the
// result must be the exact ranking.
func TestHNSWIndex_DifferentialExact(t *testing.T) {
	for _, metric := range []core.Metric{core.Euclidean, core.Cosine} {
		t.Run(metric.Name(), func(t *testing.T) {
			factory := testutil.Factory{
				New: func() core.Index {
					index, err := hnsw.New(16, hnsw.WithM(16), hnsw.WithEf(100), hnsw.WithMetric(metric))
					if err != nil {
						panic(fmt.Sprintf("hnsw.New failed: %v", err))
					}
					return index
				},
				ExactDistances: true,
				SortedResults:  true,
				Metric:         metric,
			}
			testutil.RunExactDifferential(t, factory, 16, 300, 10)
		})
	}
}

// TestHNSWIndex_DifferentialBulkSequential compares an index built through
// Add and Delete with one built through BulkAdd and BulkDelete.
func TestHNSWIndex_DifferentialBulkSequential(t *testing.T) {
	testutil.RunBulkSequentialDifferential(t, hnswFactory(), 16, 300, 10)
}

// TestHNSWIndex_DifferentialUpdate compares an index whose vectors were moved
// through Update and BulkUpdate with one built directly from the final data.
func TestHNSWIndex_DifferentialUpdate(t *testing.T) {
	testutil.RunUpdateDifferential(t, hnswFactory(), 16, 300, 10)
}

// TestHNSWIndex_DifferentialSaveLoad compares complete searches before and
// after a save and load round-trip of an index that has seen deletions.
func TestHNSWIndex_DifferentialSaveLoad(t *testing.T) {
	testutil.RunSaveLoadDifferential(t, hnswFactory(), 16, 300, 10)
}

// TestHNSWIndex_FallbackReturnsTrueTopK checks that a search that falls back
// to the brute-force scan returns the exact k nearest neighbors. The scan
// reads every stored vector, so nothing less is acceptable. A tiny ef forces
// the graph to return fewer than k candidates on every query.
func TestHNSWIndex_FallbackReturnsTrueTopK(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim, n, k := 16, 2000, 10
	idx := newTestIndex(t, dim, hnsw.WithM(4), hnsw.WithEf(2))
	rng := rand.New(rand.NewSource(2))
	data := make(map[int][]float32, n)
	for id := 0; id < n; id++ {
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		data[id] = vec
	}
	if err := idx.BulkAdd(data); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	for qi := 0; qi < 50; qi++ {
		query := make([]float32, dim)
		for i := range query {
			query[i] = float32(rng.NormFloat64())
		}
		want, err := testutil.BruteForceKNN(query, data, k, core.Euclidean)
		if err != nil {
			t.Fatalf("query %d: BruteForceKNN failed: %v", qi, err)
		}
		got, err := idx.Search(query, k)
		if err != nil {
			t.Fatalf("query %d: Search failed: %v", qi, err)
		}
		if recall := testutil.Recall(got, want); recall < 1.0 {
			t.Errorf("query %d: fallback search returned recall %.2f, want 1.0", qi, recall)
		}
	}
	if fb := idx.Stats().FallbackSearches; fb < 50 {
		t.Fatalf("expected every query to use the fallback, got %d of 50", fb)
	}
}

// TestHNSWIndex_BulkAddLargeExactness bulk-adds enough vectors to spread
// across many insertion workers, then requires complete searches to return
// the exact brute-force ranking. This guards the concurrent insertion path:
// a lost edge cannot hide, because a complete search covers every node, and
// a corrupted link structure surfaces as a wrong ranking or a wrong count.
func TestHNSWIndex_BulkAddLargeExactness(t *testing.T) {
	testutil.RunExactDifferential(t, hnswFactory(), 16, 4000, 5)
}

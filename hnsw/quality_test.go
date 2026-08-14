package hnsw_test

import (
	"fmt"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/hnsw"
	"github.com/habedi/hann/internal/testutil"
)

// measureMeanRecall builds indexes over clustered data with the given metric
// and returns the mean recall at k=10 over 20 queries per build against the
// brute-force ground truth. The graph shape depends on the package-level
// level generator, so recall varies between builds; averaging over three
// builds narrows the spread the assertion has to allow for.
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
		// The index normalizes cosine vectors in place, so it gets copies and
		// the ground truth keeps the raw data.
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
	// selection, because it keeps the level 0 graph connected across
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
// must reach every node through the graph alone, so the brute-force fallback
// counter has to stay at zero. Closest-M neighbor selection used to split the
// graph into per-cluster components, which this test would catch.
func TestHNSWIndex_Level0Connectivity(t *testing.T) {
	const (
		dim      = 16
		n        = 2000
		clusters = 16
	)
	data := testutil.ClusteredData(42, n, dim, clusters)
	index := newTestIndex(t, dim, hnsw.WithM(16), hnsw.WithEf(n))
	vectors := make(map[int][]float32, len(data))
	for id, vec := range data {
		vectors[id] = testutil.CopyVector(vec)
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	query := testutil.Queries(43, data, 1)[0]
	results, err := index.Search(testutil.CopyVector(query), n)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != n {
		t.Errorf("search with k=%d returned %d results", n, len(results))
	}
	if fb := index.Stats().FallbackSearches; fb != 0 {
		t.Errorf("graph search reached only part of the index: %d fallback searches, want 0", fb)
	}
}

// TestHNSWIndex_DifferentialExact compares complete searches against brute
// force for both supported distances; with k equal to the index size the
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

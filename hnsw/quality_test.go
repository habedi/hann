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
	// Observed range over 40 runs: 0.64 to 0.93. The spread comes from the
	// level generator: the level 0 graph splits into per-cluster components,
	// so recall depends on how well the upper levels route between clusters.
	// The threshold is a regression tripwire, not a quality goal.
	if recall < 0.50 {
		t.Errorf("euclidean mean recall %.4f is below the regression threshold 0.50", recall)
	}
}

func TestHNSWIndex_RecallCosine(t *testing.T) {
	recall := measureMeanRecall(t, core.Cosine)
	t.Logf("cosine mean recall at k=10: %.4f", recall)
	// Observed range over 40 runs: 0.55 to 0.92, with more spread than the
	// euclidean variant. The threshold is a regression tripwire, not a
	// quality goal.
	if recall < 0.40 {
		t.Errorf("cosine mean recall %.4f is below the regression threshold 0.40", recall)
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

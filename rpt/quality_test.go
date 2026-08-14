package rpt_test

import (
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/internal/testutil"
)

// TestRPTIndex_Recall measures recall on clustered synthetic data against a
// brute-force ground truth. With HANN_SEED fixed the tree build is
// reproducible, so the measured recall is stable across runs. Observed recall
// over repeated runs was 1.00 for every query batch; the threshold of 0.90
// leaves a margin well above 0.05 for platform differences in the SIMD path.
func TestRPTIndex_Recall(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim, k := 16, 10
	data := testutil.ClusteredData(42, 2000, dim, 16)
	queries := testutil.Queries(43, data, 20)

	idx := mustNew(t, dim)
	arg := make(map[int][]float32, len(data))
	for id, vec := range data {
		arg[id] = testutil.CopyVector(vec)
	}
	if err := idx.BulkAdd(arg); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	var total float64
	for i, query := range queries {
		want, err := testutil.BruteForceKNN(query, data, k, core.Euclidean)
		if err != nil {
			t.Fatalf("BruteForceKNN for query %d failed: %v", i, err)
		}
		got, err := idx.Search(testutil.CopyVector(query), k)
		if err != nil {
			t.Fatalf("Search for query %d failed: %v", i, err)
		}
		total += testutil.Recall(got, want)
	}
	recall := total / float64(len(queries))
	t.Logf("mean recall over %d queries at k=%d: %.4f", len(queries), k, recall)
	if recall < 0.90 {
		t.Errorf("mean recall %.4f is below the threshold 0.90", recall)
	}
}

// TestRPTIndex_DifferentialExact compares complete searches against brute
// force; with k equal to the index size the result must be the exact ranking.
func TestRPTIndex_DifferentialExact(t *testing.T) {
	testutil.RunExactDifferential(t, rptFactory(t, 16), 16, 300, 10)
}

// TestRPTIndex_DifferentialBulkSequential compares an index built through Add
// and Delete with one built through BulkAdd and BulkDelete.
func TestRPTIndex_DifferentialBulkSequential(t *testing.T) {
	testutil.RunBulkSequentialDifferential(t, rptFactory(t, 16), 16, 300, 10)
}

package rpt_test

import (
	"math/rand"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/internal/testutil"
	"github.com/habedi/hann/rpt"
)

// TestRPTIndex_Recall measures recall on clustered synthetic data against a
// brute-force ground truth. With HANN_SEED fixed the tree build is
// reproducible, so the measured recall is stable across runs. The observed
// recall with the default parameters was 0.93. The threshold of 0.90 leaves
// room for platform differences in the SIMD path.
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

// TestRPTIndex_TreeSelfConsistency checks that a query equal to a stored
// vector is routed to the leaf that holds it. The query projects to the same
// value as the stored point at every node, so routing must reach the same
// leaf. A tree whose split thresholds disagree with the assignment of points
// to children fails this. The leaf capacity is set above k and the
// brute-force fallback is disabled, so the answer must come from the tree.
func TestRPTIndex_TreeSelfConsistency(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim, n := 64, 2000
	rng := rand.New(rand.NewSource(7))
	data := make(map[int][]float32, n)
	for id := 0; id < n; id++ {
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		data[id] = vec
	}

	idx := mustNew(t, dim, rpt.WithLeafCapacity(32), rpt.WithBruteForceFallback(false))
	if err := idx.BulkAdd(data); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	misses := 0
	for id := 0; id < 100; id++ {
		got, err := idx.Search(testutil.CopyVector(data[id]), 1)
		if err != nil {
			t.Fatalf("Search for id %d failed: %v", id, err)
		}
		if len(got) == 0 || got[0].ID != id {
			misses++
		}
	}
	if misses > 0 {
		t.Errorf("%d of 100 self queries did not return the queried id", misses)
	}
	if fallbacks := idx.Stats().FallbackSearches; fallbacks != 0 {
		t.Errorf("expected no fallback searches, got %d", fallbacks)
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

// TestRPTIndex_DifferentialUpdate compares an index whose vectors were moved
// through Update and BulkUpdate with one built directly from the final data.
func TestRPTIndex_DifferentialUpdate(t *testing.T) {
	testutil.RunUpdateDifferential(t, rptFactory(t, 16), 16, 300, 10)
}

// TestRPTIndex_DifferentialSaveLoad compares complete searches before and
// after a save and load round-trip of an index that has seen deletions.
func TestRPTIndex_DifferentialSaveLoad(t *testing.T) {
	testutil.RunSaveLoadDifferential(t, rptFactory(t, 16), 16, 300, 10)
}

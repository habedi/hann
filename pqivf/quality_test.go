package pqivf_test

import (
	"fmt"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/internal/testutil"
)

func TestPQIVF_SyntheticRecall(t *testing.T) {
	const (
		n        = 2000
		dim      = 16
		clusters = 16
		q        = 20
		k        = 10
	)
	data := testutil.ClusteredData(42, n, dim, clusters)

	idx := newIndex(t, dim, 16, 2, 16, 10)
	arg := make(map[int][]float32, len(data))
	for id, vec := range data {
		arg[id] = testutil.CopyVector(vec)
	}
	if err := idx.BulkAdd(arg); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	queries := testutil.Queries(43, data, q)
	total := 0.0
	for _, query := range queries {
		want, err := testutil.BruteForceKNN(query, data, k, core.Euclidean)
		if err != nil {
			t.Fatalf("BruteForceKNN failed: %v", err)
		}
		got, err := idx.Search(testutil.CopyVector(query), k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		total += testutil.Recall(got, want)
	}
	recall := total / float64(q)
	t.Logf("PQIVF recall@%d over %d queries: %.3f", k, q, recall)

	// PQ quantization keeps recall well below 1 on this data. Observed
	// recall over seven runs ranged from 0.205 to 0.300, so the threshold
	// sits well below the minimum to absorb k-means seeding variance.
	const threshold = 0.10
	if recall < threshold {
		t.Fatalf("recall %.3f is below the threshold %.3f", recall, threshold)
	}
}

// TestPQIVF_DifferentialBulkSequential compares an index built through Add
// and Delete with one built through BulkAdd and BulkDelete. Distances are
// quantized and the two indexes train independently, so only the id sets and
// the counts are compared, not the rankings.
func TestPQIVF_DifferentialBulkSequential(t *testing.T) {
	factory := testutil.Factory{
		New: func() core.Index {
			return newIndex(t, 16, 4, 2, 8, 10)
		},
		Train: func(idx core.Index) error {
			trainer, ok := idx.(core.Trainer)
			if !ok {
				return fmt.Errorf("index does not implement core.Trainer")
			}
			return trainer.Train()
		},
		MinTrainSize:   4,
		ExactDistances: false,
		SortedResults:  true,
		Metric:         core.Euclidean,
	}
	testutil.RunBulkSequentialDifferential(t, factory, 16, 300, 10)
}

// TestPQIVF_DifferentialUpdate compares an index whose vectors were moved
// through Update and BulkUpdate with one built directly from the final data.
// Distances are quantized, so the check covers the id bookkeeping: every
// surviving id must be returned exactly once by a complete search.
func TestPQIVF_DifferentialUpdate(t *testing.T) {
	testutil.RunUpdateDifferential(t, pqivfFactory(t), 16, 300, 10)
}

// TestPQIVF_DifferentialSaveLoad compares complete searches before and after
// a save and load round-trip. The codebooks travel with the file, so the
// loaded index must reproduce the original distances rank by rank even
// though they are quantized.
func TestPQIVF_DifferentialSaveLoad(t *testing.T) {
	testutil.RunSaveLoadDifferential(t, pqivfFactory(t), 16, 300, 10)
}

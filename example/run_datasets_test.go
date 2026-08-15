package example_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/pqivf"
	"github.com/habedi/hann/rpt"
)

// TestRunDataset covers the success paths of RunDataset. It uses a synthetic
// dataset written to t.TempDir(). The failure paths (a missing dataset, a
// failing BulkAdd, a failing Train, or a failing Search) call log.Fatalf,
// which exits the test process. So they cannot be tested here. The loader
// tests cover them indirectly.
func TestRunDataset(t *testing.T) {
	t.Setenv("HANN_SEED", "42")

	factory := func() core.Index {
		index, err := rpt.New(4)
		if err != nil {
			t.Fatalf("failed to create RPT index: %v", err)
		}
		return index
	}

	root := t.TempDir()
	dataset := "synthetic"
	datasetDir := filepath.Join(root, dataset)
	if err := os.MkdirAll(datasetDir, 0o755); err != nil {
		t.Fatalf("failed to create dataset directory: %v", err)
	}
	writeDataset(t, datasetDir)

	t.Run("per-query output mode", func(t *testing.T) {
		// numQueries equals the number of test vectors, so RunDataset
		// prints per-query details instead of a progress bar.
		example.RunDataset(factory, dataset, root, 2, 2, 3)
	})

	t.Run("benchmark mode with worker threads", func(t *testing.T) {
		// A negative numQueries switches RunDataset to benchmark mode,
		// and HANN_BENCH_NTRD sets the worker count.
		t.Setenv("HANN_BENCH_NTRD", "2")
		example.RunDataset(factory, dataset, root, 2, -1, 3)
	})

	t.Run("index with a training step", func(t *testing.T) {
		// A PQIVF index implements core.Trainer, so this covers the
		// branch that trains the index before searching. The cluster
		// counts are kept small so six vectors are enough to train.
		trainedFactory := func() core.Index {
			index, err := pqivf.New(4,
				pqivf.WithCoarseK(2),
				pqivf.WithNumSubquantizers(2),
				pqivf.WithPQK(2))
			if err != nil {
				t.Fatalf("failed to create PQIVF index: %v", err)
			}
			return index
		}
		example.RunDataset(trainedFactory, dataset, root, 2, 2, 3)
	})
}

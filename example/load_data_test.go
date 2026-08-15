package example_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/habedi/hann/example"
	"github.com/habedi/hann/rpt"
)

// The fixtures below are tiny synthetic datasets written to t.TempDir(),
// so the tests do not depend on the downloaded example datasets.

// trainRows holds six 4-dimensional training vectors (ids 0 through 5).
var trainRows = []string{
	"0,0,0,0",
	"1,0,0,0",
	"0,1,0,0",
	"10,10,10,10",
	"10,10,10,11",
	"5,5,5,5",
}

// testRows holds two query vectors. Under Euclidean distance, the nearest
// training vectors to the first query are ids 0 and 1, and the nearest to
// the second query are ids 3 and 4.
var testRows = []string{
	"0.1,0,0,0",
	"10,10,10,10.2",
}

var neighborRows = []string{
	"0,1",
	"3,4",
}

var distanceRows = []string{
	"0.1,0.9",
	"0.2,0.8",
}

// writeFile writes lines to a file inside dir and fails the test on error.
func writeFile(t *testing.T, dir, name string, lines []string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("failed to write %s: %v", name, err)
	}
	return path
}

// writeDataset writes a complete dataset directory (train.csv, test.csv,
// neighbors.csv, and distances.csv) and returns its path.
func writeDataset(t *testing.T, dir string) string {
	t.Helper()
	writeFile(t, dir, "train.csv", trainRows)
	writeFile(t, dir, "test.csv", testRows)
	writeFile(t, dir, "neighbors.csv", neighborRows)
	writeFile(t, dir, "distances.csv", distanceRows)
	return dir
}

// newTestIndex creates a small RPT index for the 4-dimensional fixtures.
func newTestIndex(t *testing.T) *rpt.Index {
	t.Helper()
	index, err := rpt.New(4)
	if err != nil {
		t.Fatalf("failed to create RPT index: %v", err)
	}
	return index
}

func TestLoadDataset(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dir := writeDataset(t, t.TempDir())
	index := newTestIndex(t)

	testVectors, trueNeighbors, trueDistances, err := example.LoadDataset(index, dir)
	if err != nil {
		t.Fatalf("LoadDataset returned an error: %v", err)
	}
	if got := index.Stats().Count; got != len(trainRows) {
		t.Errorf("expected %d indexed vectors, got %d", len(trainRows), got)
	}
	if len(testVectors) != 2 || len(testVectors[0]) != 4 {
		t.Errorf("unexpected test vectors shape: %v", testVectors)
	}
	if len(trueNeighbors) != 2 || trueNeighbors[1][0] != 3 {
		t.Errorf("unexpected ground-truth neighbors: %v", trueNeighbors)
	}
	if len(trueDistances) != 2 || trueDistances[0][0] != 0.1 {
		t.Errorf("unexpected ground-truth distances: %v", trueDistances)
	}

	// The loaded index should find the hand-computed nearest neighbor for
	// each query. The dataset is small enough for the tree to be exact.
	for i, query := range testVectors {
		results, err := index.Search(query, 2)
		if err != nil {
			t.Fatalf("search failed on query %d: %v", i, err)
		}
		if len(results) == 0 || results[0].ID != trueNeighbors[i][0] {
			t.Errorf("query %d: expected nearest id %d, got %v",
				i, trueNeighbors[i][0], results)
		}
	}
}

func TestLoadDatasetErrors(t *testing.T) {
	// Each subtest removes one file from a complete dataset directory so
	// that a specific loading step fails.
	files := []string{"train.csv", "test.csv", "neighbors.csv", "distances.csv"}
	for _, missing := range files {
		t.Run("missing "+missing, func(t *testing.T) {
			dir := writeDataset(t, t.TempDir())
			if err := os.Remove(filepath.Join(dir, missing)); err != nil {
				t.Fatalf("failed to remove %s: %v", missing, err)
			}
			_, _, _, err := example.LoadDataset(newTestIndex(t), dir)
			if err == nil {
				t.Fatalf("expected an error when %s is missing", missing)
			}
			if !strings.Contains(err.Error(), missing) {
				t.Errorf("expected error to mention %s, got: %v", missing, err)
			}
		})
	}
}

func TestLoadCSV(t *testing.T) {
	t.Run("happy path", func(t *testing.T) {
		path := writeFile(t, t.TempDir(), "train.csv", trainRows)
		index := newTestIndex(t)
		if err := example.LoadCSV(index, path, false); err != nil {
			t.Fatalf("LoadCSV returned an error: %v", err)
		}
		if got := index.Stats().Count; got != len(trainRows) {
			t.Errorf("expected %d vectors in the index, got %d", len(trainRows), got)
		}
	})

	t.Run("skip header", func(t *testing.T) {
		rows := append([]string{"a,b,c,d"}, trainRows...)
		// The header contains non-numeric values, so it must be skipped
		// for the load to succeed.
		path := writeFile(t, t.TempDir(), "train.csv", rows)
		index := newTestIndex(t)
		if err := example.LoadCSV(index, path, true); err != nil {
			t.Fatalf("LoadCSV with skipHeader returned an error: %v", err)
		}
		if got := index.Stats().Count; got != len(trainRows) {
			t.Errorf("expected %d vectors in the index, got %d", len(trainRows), got)
		}
	})

	t.Run("missing file", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "absent.csv")
		if err := example.LoadCSV(newTestIndex(t), path, false); err == nil {
			t.Fatal("expected an error for a missing file")
		}
	})

	t.Run("non-numeric value", func(t *testing.T) {
		path := writeFile(t, t.TempDir(), "train.csv", []string{"0,0,abc,0"})
		err := example.LoadCSV(newTestIndex(t), path, false)
		if err == nil {
			t.Fatal("expected a parse error for a non-numeric value")
		}
		if !strings.Contains(err.Error(), "parse error") {
			t.Errorf("expected a parse error, got: %v", err)
		}
	})

	t.Run("inconsistent column count", func(t *testing.T) {
		path := writeFile(t, t.TempDir(), "train.csv", []string{"0,0,0,0", "1,2"})
		err := example.LoadCSV(newTestIndex(t), path, false)
		if err == nil {
			t.Fatal("expected an error for rows with different column counts")
		}
		if !strings.Contains(err.Error(), "read error") {
			t.Errorf("expected a read error, got: %v", err)
		}
	})

	t.Run("dimension mismatch on add", func(t *testing.T) {
		// The index expects dimension 4, so 3-column rows make Add fail.
		path := writeFile(t, t.TempDir(), "train.csv", []string{"0,0,0"})
		err := example.LoadCSV(newTestIndex(t), path, false)
		if err == nil {
			t.Fatal("expected an error when the vector dimension does not match")
		}
		if !strings.Contains(err.Error(), "failed to add vector 0") {
			t.Errorf("expected an add error for vector 0, got: %v", err)
		}
	})
}

func TestLoadTrainingVectors(t *testing.T) {
	t.Run("happy path", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "train.csv", trainRows)
		vectors, err := example.LoadTrainingVectors(dir)
		if err != nil {
			t.Fatalf("LoadTrainingVectors returned an error: %v", err)
		}
		if len(vectors) != len(trainRows) {
			t.Fatalf("expected %d vectors, got %d", len(trainRows), len(vectors))
		}
		want := []float32{10, 10, 10, 11}
		got := vectors[4]
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("vector 4: expected %v, got %v", want, got)
				break
			}
		}
	})

	t.Run("missing file", func(t *testing.T) {
		if _, err := example.LoadTrainingVectors(t.TempDir()); err == nil {
			t.Fatal("expected an error when train.csv is missing")
		}
	})
}

func TestLoadTestDataset(t *testing.T) {
	t.Run("happy path", func(t *testing.T) {
		dir := writeDataset(t, t.TempDir())
		testVectors, neighbors, distances, err := example.LoadTestDataset(dir)
		if err != nil {
			t.Fatalf("LoadTestDataset returned an error: %v", err)
		}
		if len(testVectors) != 2 || testVectors[0][0] != 0.1 {
			t.Errorf("unexpected test vectors: %v", testVectors)
		}
		if len(neighbors) != 2 || neighbors[0][1] != 1 {
			t.Errorf("unexpected neighbors: %v", neighbors)
		}
		if len(distances) != 2 || distances[1][1] != 0.8 {
			t.Errorf("unexpected distances: %v", distances)
		}
	})

	// Each subtest removes one file so a specific loading step fails.
	files := []string{"test.csv", "neighbors.csv", "distances.csv"}
	for _, missing := range files {
		t.Run("missing "+missing, func(t *testing.T) {
			dir := writeDataset(t, t.TempDir())
			if err := os.Remove(filepath.Join(dir, missing)); err != nil {
				t.Fatalf("failed to remove %s: %v", missing, err)
			}
			_, _, _, err := example.LoadTestDataset(dir)
			if err == nil {
				t.Fatalf("expected an error when %s is missing", missing)
			}
			if !strings.Contains(err.Error(), missing) {
				t.Errorf("expected error to mention %s, got: %v", missing, err)
			}
		})
	}

	// parseValue rejections surface through the loaders: an integer column
	// rejects a float, and a float column rejects a non-numeric token.
	t.Run("non-integer neighbor id", func(t *testing.T) {
		dir := writeDataset(t, t.TempDir())
		writeFile(t, dir, "neighbors.csv", []string{"0,1.5"})
		if _, _, _, err := example.LoadTestDataset(dir); err == nil {
			t.Fatal("expected a parse error for a non-integer neighbor id")
		}
	})

	t.Run("non-numeric distance", func(t *testing.T) {
		dir := writeDataset(t, t.TempDir())
		writeFile(t, dir, "distances.csv", []string{"0.1,oops"})
		if _, _, _, err := example.LoadTestDataset(dir); err == nil {
			t.Fatal("expected a parse error for a non-numeric distance")
		}
	})

	// parseValue trims surrounding whitespace before parsing.
	t.Run("whitespace around values", func(t *testing.T) {
		dir := writeDataset(t, t.TempDir())
		writeFile(t, dir, "distances.csv", []string{"0.1, 0.9", "0.2, 0.8"})
		_, _, distances, err := example.LoadTestDataset(dir)
		if err != nil {
			t.Fatalf("expected whitespace to be trimmed, got error: %v", err)
		}
		if distances[0][1] != 0.9 {
			t.Errorf("expected 0.9 after trimming, got %v", distances[0][1])
		}
	})
}

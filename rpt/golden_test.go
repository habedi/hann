package rpt_test

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/internal/testutil"
	"github.com/habedi/hann/rpt"
)

// update regenerates the golden files under testdata when set. Run
// "go test ./rpt/ -run GoldenFile -update" to regenerate them after an
// intentional format change.
var update = flag.Bool("update", false, "regenerate the golden files under testdata")

// goldenExpected is the JSON shape stored next to the golden gob fixture.
type goldenExpected struct {
	Count   int     `json:"count"`
	Results [][]int `json:"results"`
}

// goldenSearchIDs returns the result ids of Search on the fixed golden
// queries.
func goldenSearchIDs(t *testing.T, idx *rpt.Index, queries [][]float32, k int) [][]int {
	t.Helper()
	results := make([][]int, len(queries))
	for i, query := range queries {
		neighbors, err := idx.Search(testutil.CopyVector(query), k)
		if err != nil {
			t.Fatalf("Search for golden query %d failed: %v", i, err)
		}
		ids := make([]int, len(neighbors))
		for j, n := range neighbors {
			ids[j] = n.ID
		}
		results[i] = ids
	}
	return results
}

// TestRPTIndex_GoldenFile pins the gob on-disk format. The serialized form
// holds only the points, not the tree, so Load rebuilds the tree. That
// rebuild draws randomness. This is why both the
// update path and the verify path run under a fixed HANN_SEED. The loading
// index is constructed with the Manhattan metric. The test therefore also
// pins that the metric name stored in the file overrides the configured one.
// Run with -update to regenerate the fixture after an intentional format
// change. The test never regenerates it on its own.
func TestRPTIndex_GoldenFile(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	const (
		dim = 8
		k   = 5
	)
	gobPath := filepath.Join("testdata", "index_v1.gob")
	jsonPath := filepath.Join("testdata", "index_v1_expected.json")
	data := testutil.ClusteredData(7, 30, dim, 4)
	queries := testutil.Queries(8, data, 3)

	newIndex := func() *rpt.Index {
		return mustNew(t, dim, rpt.WithMetric(core.Manhattan))
	}

	if *update {
		idx := mustNew(t, dim)
		arg := make(map[int][]float32, len(data))
		for id, vec := range data {
			arg[id] = testutil.CopyVector(vec)
		}
		if err := idx.BulkAdd(arg); err != nil {
			t.Fatalf("BulkAdd failed: %v", err)
		}
		if err := idx.Delete(0); err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatalf("creating testdata failed: %v", err)
		}
		f, err := os.Create(gobPath)
		if err != nil {
			t.Fatalf("creating %s failed: %v", gobPath, err)
		}
		if err := idx.Save(f); err != nil {
			t.Fatalf("Save failed: %v", err)
		}
		if err := f.Close(); err != nil {
			t.Fatalf("closing %s failed: %v", gobPath, err)
		}
		// Compute the expected values from a freshly loaded index. That is
		// the exact state the verify path checks.
		loaded := newIndex()
		g, err := os.Open(gobPath)
		if err != nil {
			t.Fatalf("opening %s failed: %v", gobPath, err)
		}
		if err := loaded.Load(g); err != nil {
			t.Fatalf("Load failed: %v", err)
		}
		if err := g.Close(); err != nil {
			t.Fatalf("closing %s failed: %v", gobPath, err)
		}
		expected := goldenExpected{
			Count:   loaded.Stats().Count,
			Results: goldenSearchIDs(t, loaded, queries, k),
		}
		out, err := json.MarshalIndent(expected, "", "  ")
		if err != nil {
			t.Fatalf("marshaling expected values failed: %v", err)
		}
		if err := os.WriteFile(jsonPath, append(out, '\n'), 0o644); err != nil {
			t.Fatalf("writing %s failed: %v", jsonPath, err)
		}
		t.Logf("regenerated %s and %s", gobPath, jsonPath)
		return
	}

	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("reading %s failed (run with -update to generate it): %v", jsonPath, err)
	}
	var expected goldenExpected
	if err := json.Unmarshal(raw, &expected); err != nil {
		t.Fatalf("unmarshaling %s failed: %v", jsonPath, err)
	}

	g, err := os.Open(gobPath)
	if err != nil {
		t.Fatalf("opening %s failed (run with -update to generate it): %v", gobPath, err)
	}
	defer func() {
		if err := g.Close(); err != nil {
			t.Errorf("closing %s failed: %v", gobPath, err)
		}
	}()
	loaded := newIndex()
	if err := loaded.Load(g); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// The fixture was written with the Euclidean metric, so the loaded index
	// must report the name stored in the file, not the configured one.
	if got := loaded.Stats().Distance; got != "euclidean" {
		t.Errorf("Stats().Distance = %q after load, golden file expects %q", got, "euclidean")
	}
	if got := loaded.Stats().Count; got != expected.Count {
		t.Errorf("Stats().Count = %d, golden file expects %d", got, expected.Count)
	}
	results := goldenSearchIDs(t, loaded, queries, k)
	for i, want := range expected.Results {
		got := results[i]
		if len(got) != len(want) {
			t.Errorf("query %d: got %d results, golden file expects %d", i, len(got), len(want))
			continue
		}
		for j := range want {
			if got[j] != want[j] {
				t.Errorf("query %d result %d: got id %d, golden file expects id %d",
					i, j, got[j], want[j])
			}
		}
	}
}

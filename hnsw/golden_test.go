package hnsw_test

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/habedi/hann/hnsw"
	"github.com/habedi/hann/internal/testutil"
)

var update = flag.Bool("update", false, "regenerate golden files")

// goldenExpected pins the observable behavior of the golden index file:
// the stored vector count and the ids returned for three fixed queries.
type goldenExpected struct {
	Count     int     `json:"count"`
	ResultIDs [][]int `json:"result_ids"`
}

// goldenData returns the deterministic vectors and queries the golden fixture
// is built from and queried with.
func goldenData() (map[int][]float32, [][]float32) {
	data := testutil.ClusteredData(7, 30, 8, 4)
	queries := testutil.Queries(8, data, 3)
	return data, queries
}

// TestHNSWIndex_GoldenFile pins the gob on-disk format. In normal mode it
// loads a committed fixture and checks the stats and search results against
// the committed expectations; it must never regenerate the fixture
// implicitly, because a fixture written by an older version has to keep
// loading. Run with -update to rebuild the fixture after a deliberate,
// backward-compatible format change.
func TestHNSWIndex_GoldenFile(t *testing.T) {
	gobPath := filepath.Join("testdata", "index_v1.gob")
	jsonPath := filepath.Join("testdata", "index_v1_expected.json")
	data, queries := goldenData()

	if *update {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatalf("failed to create testdata: %v", err)
		}
		index := newTestIndex(t, 8, hnsw.WithM(8), hnsw.WithEf(50))
		// Insert id 0 first, so it can plausibly become the entry point, which
		// is the case the HasEntryPoint flag exists for.
		for id := 0; id < len(data); id++ {
			if err := index.Add(id, testutil.CopyVector(data[id])); err != nil {
				t.Fatalf("Add(%d) failed: %v", id, err)
			}
		}
		if err := index.Delete(5); err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		f, err := os.Create(gobPath)
		if err != nil {
			t.Fatalf("failed to create %s: %v", gobPath, err)
		}
		if err := index.Save(f); err != nil {
			t.Fatalf("Save failed: %v", err)
		}
		if err := f.Close(); err != nil {
			t.Fatalf("failed to close %s: %v", gobPath, err)
		}
		expected := goldenExpected{Count: index.Stats().Count}
		for _, query := range queries {
			neighbors, err := index.Search(testutil.CopyVector(query), 5)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}
			ids := make([]int, len(neighbors))
			for i, n := range neighbors {
				ids[i] = n.ID
			}
			expected.ResultIDs = append(expected.ResultIDs, ids)
		}
		out, err := json.MarshalIndent(expected, "", "  ")
		if err != nil {
			t.Fatalf("failed to marshal expectations: %v", err)
		}
		if err := os.WriteFile(jsonPath, append(out, '\n'), 0o644); err != nil {
			t.Fatalf("failed to write %s: %v", jsonPath, err)
		}
		t.Logf("regenerated %s and %s", gobPath, jsonPath)
		return
	}

	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("failed to read %s (run with -update to generate it): %v", jsonPath, err)
	}
	var expected goldenExpected
	if err := json.Unmarshal(raw, &expected); err != nil {
		t.Fatalf("failed to parse %s: %v", jsonPath, err)
	}
	f, err := os.Open(gobPath)
	if err != nil {
		t.Fatalf("failed to open %s (run with -update to generate it): %v", gobPath, err)
	}
	defer f.Close()
	loaded := newTestIndex(t, 8, hnsw.WithM(8), hnsw.WithEf(50))
	if err := loaded.Load(f); err != nil {
		t.Fatalf("Load failed on the golden fixture: %v", err)
	}
	if got := loaded.Stats().Count; got != expected.Count {
		t.Errorf("Stats().Count = %d after loading the golden fixture, want %d", got, expected.Count)
	}
	for qi, query := range queries {
		neighbors, err := loaded.Search(testutil.CopyVector(query), 5)
		if err != nil {
			t.Fatalf("Search failed on the loaded golden fixture: %v", err)
		}
		ids := make([]int, len(neighbors))
		for i, n := range neighbors {
			ids[i] = n.ID
		}
		want := expected.ResultIDs[qi]
		if len(ids) != len(want) {
			t.Errorf("query %d returned ids %v, want %v", qi, ids, want)
			continue
		}
		for i := range ids {
			if ids[i] != want[i] {
				t.Errorf("query %d returned ids %v, want %v", qi, ids, want)
				break
			}
		}
	}
}

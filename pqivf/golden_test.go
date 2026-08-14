package pqivf_test

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/habedi/hann/internal/testutil"
)

// update regenerates the golden test fixtures when set. Run
// go test ./pqivf/ -run TestPQIVF_GoldenFile -update to refresh them.
var update = flag.Bool("update", false, "regenerate golden test fixtures")

// goldenExpected is the JSON shape of the golden fixture expectations.
type goldenExpected struct {
	Count   int     `json:"count"`
	Results [][]int `json:"results"`
}

// goldenQueries returns the fixed queries the golden test searches with.
// The data and the queries are deterministic, so the fixture and the
// verification see the same inputs.
func goldenQueries() (map[int][]float32, [][]float32) {
	data := testutil.ClusteredData(7, 30, 8, 3)
	queries := testutil.Queries(8, data, 3)
	return data, queries
}

func TestPQIVF_GoldenFile(t *testing.T) {
	gobPath := filepath.Join("testdata", "index_v1.gob")
	jsonPath := filepath.Join("testdata", "index_v1_expected.json")
	const k = 5

	data, queries := goldenQueries()

	if *update {
		idx := newIndex(t, 8, 3, 2, 4, 5)
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
		if err := idx.Delete(0); err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		if err := idx.Train(); err != nil {
			t.Fatalf("re-Train failed: %v", err)
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

		expected := goldenExpected{Count: idx.Stats().Count}
		for _, query := range queries {
			results, err := idx.Search(testutil.CopyVector(query), k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}
			ids := make([]int, len(results))
			for i, n := range results {
				ids[i] = n.ID
			}
			expected.Results = append(expected.Results, ids)
		}
		out, err := json.MarshalIndent(expected, "", "  ")
		if err != nil {
			t.Fatalf("marshaling expectations failed: %v", err)
		}
		if err := os.WriteFile(jsonPath, append(out, '\n'), 0o644); err != nil {
			t.Fatalf("writing %s failed: %v", jsonPath, err)
		}
		t.Logf("regenerated %s and %s", gobPath, jsonPath)
		return
	}

	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("reading %s failed (run with -update to generate): %v", jsonPath, err)
	}
	var expected goldenExpected
	if err := json.Unmarshal(raw, &expected); err != nil {
		t.Fatalf("unmarshaling %s failed: %v", jsonPath, err)
	}

	f, err := os.Open(gobPath)
	if err != nil {
		t.Fatalf("opening %s failed (run with -update to generate): %v", gobPath, err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			t.Errorf("closing %s failed: %v", gobPath, err)
		}
	}()
	idx := newIndex(t, 8, 3, 2, 4, 5)
	if err := idx.Load(f); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if got := idx.Stats().Count; got != expected.Count {
		t.Fatalf("Stats().Count = %d, golden fixture expects %d", got, expected.Count)
	}

	// Search twice, so the test also confirms that a loaded trained index
	// answers deterministically without retraining.
	for pass := 0; pass < 2; pass++ {
		for qi, query := range queries {
			results, err := idx.Search(testutil.CopyVector(query), k)
			if err != nil {
				t.Fatalf("pass %d: Search for query %d failed: %v", pass, qi, err)
			}
			ids := make([]int, len(results))
			for i, n := range results {
				ids[i] = n.ID
			}
			want := expected.Results[qi]
			if len(ids) != len(want) {
				t.Fatalf("pass %d: query %d returned %d ids, golden fixture expects %d", pass, qi, len(ids), len(want))
			}
			for i := range ids {
				if ids[i] != want[i] {
					t.Fatalf("pass %d: query %d result %d is id %d, golden fixture expects id %d",
						pass, qi, i, ids[i], want[i])
				}
			}
		}
	}
}

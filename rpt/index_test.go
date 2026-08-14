package rpt_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/internal/testutil"
	"github.com/habedi/hann/rpt"
)

// mustNew creates an index and fails the test when the constructor returns an
// error.
func mustNew(t *testing.T, dim int, opts ...rpt.Option) *rpt.Index {
	t.Helper()
	idx, err := rpt.New(dim, opts...)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	return idx
}

func TestRPTIndex_BasicOperations(t *testing.T) {
	dim := 6
	idx := mustNew(t, dim)

	// Test Add.
	vec1 := []float32{1, 2, 3, 4, 5, 6}
	if err := idx.Add(1, vec1); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	stats := idx.Stats()
	if stats.Count != 1 {
		t.Errorf("expected count 1, got %d", stats.Count)
	}

	// Test duplicate add returns error.
	if err := idx.Add(1, vec1); err == nil {
		t.Errorf("expected error when adding duplicate id, but got none")
	}

	// Test Update.
	vec1upd := []float32{6, 5, 4, 3, 2, 1}
	if err := idx.Update(1, vec1upd); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Test Update with wrong dimension.
	wrongDim := []float32{1, 2, 3}
	if err := idx.Update(1, wrongDim); err == nil {
		t.Errorf("expected error on update with wrong dimension, but got none")
	}

	// Test Delete.
	if err := idx.Delete(1); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	stats = idx.Stats()
	if stats.Count != 0 {
		t.Errorf("expected count 0 after delete, got %d", stats.Count)
	}

	// Test Delete on non-existing id.
	if err := idx.Delete(1); err == nil {
		t.Errorf("expected error on deleting non-existent id, but got none")
	}
}

func TestRPTIndex_Search(t *testing.T) {
	dim := 6
	idx := mustNew(t, dim)

	// Insert several vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {2, 2, 2, 2, 2, 2},
		5: {1, 2, 2, 3, 4, 5},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Test error when query dimension mismatches.
	wrongQuery := []float32{1, 2, 3}
	if _, err := idx.Search(wrongQuery, 3); err == nil {
		t.Errorf("expected error for query dimension mismatch, but got none")
	}

	query := []float32{1, 2, 3, 4, 5, 6}
	neighbors, err := idx.Search(query, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 3 {
		t.Errorf("expected 3 neighbors, got %d", len(neighbors))
	}
	// Check that an exact match exists.
	found := false
	for _, n := range neighbors {
		if n.ID == 1 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected neighbor id 1 in results")
	}
}

func TestRPTIndex_BulkOperations(t *testing.T) {
	dim := 6
	idx := mustNew(t, dim)

	// BulkAdd several vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {2, 2, 2, 2, 2, 2},
		4: {3, 3, 3, 3, 3, 3},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	stats := idx.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after BulkAdd, got %d", len(vectors), stats.Count)
	}

	// BulkUpdate: update vectors 2 and 3.
	updates := map[int][]float32{
		2: {1, 1, 1, 1, 1, 1},
		3: {4, 4, 4, 4, 4, 4},
	}
	if err := idx.BulkUpdate(updates); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	// Verify update via search.
	query := []float32{1, 1, 1, 1, 1, 1}
	neighbors, err := idx.Search(query, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	found := false
	for _, nb := range neighbors {
		if nb.ID == 2 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected neighbor id 2 after BulkUpdate, but it was not found")
	}

	// BulkDelete: remove vectors 1 and 4.
	if err := idx.BulkDelete([]int{1, 4}); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}
	stats = idx.Stats()
	expected := len(vectors) - 2
	if stats.Count != expected {
		t.Errorf("expected count %d after BulkDelete, got %d", expected, stats.Count)
	}
}

func TestRPTIndex_SaveLoad(t *testing.T) {
	dim := 6
	idx := mustNew(t, dim)
	// Insert a couple of vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Save to an in-memory buffer.
	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Create a new index and load from the buffer.
	newIdx := mustNew(t, dim)
	r := bytes.NewReader(buf.Bytes())
	if err := newIdx.Load(r); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	stats := newIdx.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after load, got %d", len(vectors), stats.Count)
	}
}

func TestRPTIndex_ErrorOnWrongVectorDimension(t *testing.T) {
	dim := 6
	idx := mustNew(t, dim)

	// Test Add with wrong vector dimension.
	wrongVec := []float32{1, 2, 3}
	if err := idx.Add(1, wrongVec); err == nil {
		t.Errorf("expected error for wrong vector dimension in Add, but got none")
	}

	// Test BulkAdd with one vector having the wrong dimension.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {1, 2, 3}, // wrong dimension
	}
	if err := idx.BulkAdd(vectors); err == nil {
		t.Errorf("expected error for wrong vector dimension in BulkAdd, but got none")
	}
}

func TestRPTIndex_EdgeCases(t *testing.T) {
	dim := 4
	idx := mustNew(t, dim)

	// Search on empty index.
	if _, err := idx.Search([]float32{1, 2, 3, 4}, 1); err != nil {
		t.Errorf("expected no error searching on an empty index, but got: %v", err)
	}

	// Search with k=0.
	if _, err := idx.Search([]float32{1, 2, 3, 4}, 0); err == nil {
		t.Error("expected error searching with k=0, but got none")
	}

	// Add a vector.
	vec1 := []float32{1, 1, 1, 1}
	if err := idx.Add(1, vec1); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search with k > number of items.
	neighbors, err := idx.Search(vec1, 5)
	if err != nil {
		t.Fatalf("Search with k > num items failed: %v", err)
	}
	if len(neighbors) != 1 {
		t.Errorf("expected 1 neighbor, got %d", len(neighbors))
	}

	// Bulk operations with non-existent IDs.
	if err := idx.BulkDelete([]int{99}); err != nil {
		t.Errorf("BulkDelete with non-existent ID should not fail, but got: %v", err)
	}
	if err := idx.BulkUpdate(map[int][]float32{99: {1, 2, 3, 4}}); err == nil {
		t.Error("expected error on BulkUpdate with non-existent ID, but got none")
	}

	// Save and load empty index.
	emptyIdx := mustNew(t, dim)
	var buf bytes.Buffer
	if err := emptyIdx.Save(&buf); err != nil {
		t.Fatalf("Save on empty index failed: %v", err)
	}
	newEmptyIdx := mustNew(t, dim)
	if err := newEmptyIdx.Load(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("Load on empty index failed: %v", err)
	}
}

// makeVector returns a deterministic vector for the given id and dimension.
func makeVector(rnd *rand.Rand, dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rnd.Float32()*2 - 1
	}
	return vec
}

// TestRPTIndex_SeedReproducibility checks that two indexes built from the same
// data with the same HANN_SEED return the same search results.
func TestRPTIndex_SeedReproducibility(t *testing.T) {
	t.Setenv("HANN_SEED", "12345")
	dim := 8
	build := func() *rpt.Index {
		idx := mustNew(t, dim,
			rpt.WithLeafCapacity(5),
			rpt.WithCandidateProjections(3),
			rpt.WithParallelThreshold(1<<30),
			rpt.WithProbeMargin(0),
			rpt.WithBruteForceFallback(false))
		rnd := rand.New(rand.NewSource(7))
		for i := 0; i < 400; i++ {
			if err := idx.Add(i, makeVector(rnd, dim)); err != nil {
				t.Fatalf("Add failed: %v", err)
			}
		}
		return idx
	}
	a := build()
	b := build()

	queryRnd := rand.New(rand.NewSource(11))
	for q := 0; q < 20; q++ {
		query := makeVector(queryRnd, dim)
		na, err := a.Search(query, 5)
		if err != nil {
			t.Fatalf("Search on first index failed: %v", err)
		}
		nb, err := b.Search(query, 5)
		if err != nil {
			t.Fatalf("Search on second index failed: %v", err)
		}
		if len(na) != len(nb) {
			t.Fatalf("query %d: result counts differ: %d vs %d", q, len(na), len(nb))
		}
		for i := range na {
			if na[i].ID != nb[i].ID {
				t.Fatalf("query %d: result %d differs: id %d vs id %d",
					q, i, na[i].ID, nb[i].ID)
			}
		}
	}
}

// TestRPTIndex_SaveLoadDistance checks that the configured metric name is
// reported by Stats, survives a save and load round-trip, and that the metric
// is restored when loading into a zero-value index.
func TestRPTIndex_SaveLoadDistance(t *testing.T) {
	dim := 4
	idx := mustNew(t, dim,
		rpt.WithLeafCapacity(4),
		rpt.WithCandidateProjections(2),
		rpt.WithParallelThreshold(100),
		rpt.WithProbeMargin(0.1),
		rpt.WithMetric(core.Manhattan))
	vectors := map[int][]float32{
		1: {1, 2, 3, 4},
		2: {4, 3, 2, 1},
		3: {1, 1, 1, 1},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	if got := idx.Stats().Distance; got != "manhattan" {
		t.Errorf("Stats reported distance %q, want %q", got, "manhattan")
	}

	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded := &rpt.Index{}
	if err := loaded.Load(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if got := loaded.Stats().Distance; got != "manhattan" {
		t.Errorf("Stats after load reported distance %q, want %q", got, "manhattan")
	}
	neighbors, err := loaded.Search([]float32{1, 2, 3, 4}, 2)
	if err != nil {
		t.Fatalf("Search after load failed: %v", err)
	}
	if len(neighbors) != 2 {
		t.Errorf("expected 2 neighbors after load, got %d", len(neighbors))
	}
	if neighbors[0].ID != 1 {
		t.Errorf("expected nearest neighbor id 1, got %d", neighbors[0].ID)
	}
}

// rptFactory returns the testutil factory for the RPT index with the default
// parameters used across these tests.
func rptFactory(t *testing.T, dim int) testutil.Factory {
	return testutil.Factory{
		New: func() core.Index {
			return mustNew(t, dim)
		},
		ExactDistances: true,
		SortedResults:  true,
		Metric:         core.Euclidean,
	}
}

// TestRPTIndex_ConstructorValidation checks that the constructor rejects
// parameter values that would break the index.
func TestRPTIndex_ConstructorValidation(t *testing.T) {
	cases := []struct {
		name string
		dim  int
		opts []rpt.Option
	}{
		{"non-positive dimension", 0, nil},
		{"non-positive leaf capacity", 4, []rpt.Option{rpt.WithLeafCapacity(0)}},
		{"non-positive candidate projections", 4, []rpt.Option{rpt.WithCandidateProjections(0)}},
		{"non-positive parallel threshold", 4, []rpt.Option{rpt.WithParallelThreshold(0)}},
		{"negative probe margin", 4, []rpt.Option{rpt.WithProbeMargin(-0.1)}},
		{"zero metric", 4, []rpt.Option{rpt.WithMetric(core.Metric{})}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			idx, err := rpt.New(tc.dim, tc.opts...)
			if err == nil {
				t.Errorf("expected error from New, but got none")
			}
			if idx != nil {
				t.Errorf("expected nil index on error, got %v", idx)
			}
		})
	}
}

// TestRPTIndex_FallbackSearchCounter checks that a search that falls back to
// a brute-force scan is visible in Stats.
func TestRPTIndex_FallbackSearchCounter(t *testing.T) {
	idx := mustNew(t, 4,
		rpt.WithLeafCapacity(2),
		rpt.WithCandidateProjections(2),
		rpt.WithParallelThreshold(100),
		rpt.WithProbeMargin(0.0))
	for id := 0; id < 30; id++ {
		vec := []float32{float32(id), float32(id + 1), float32(id + 2), float32(id + 3)}
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}
	if got := idx.Stats().FallbackSearches; got != 0 {
		t.Fatalf("expected 0 fallback searches before any search, got %d", got)
	}
	// A tiny leaf capacity with k close to the point count forces the fallback.
	if _, err := idx.Search([]float32{0, 1, 2, 3}, 29); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if got := idx.Stats().FallbackSearches; got < 1 {
		t.Fatalf("expected at least 1 fallback search, got %d", got)
	}
}

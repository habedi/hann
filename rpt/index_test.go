package rpt_test

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math/rand"
	"strings"
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

	// Update vectors 2 and 3 through BulkUpdate.
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

	// Remove vectors 1 and 4 through BulkDelete.
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

// amortizedTestIndex builds an index whose search results depend strongly on
// the tree structure. It uses no probe margin and no brute-force fallback.
// A rebuild under a different seed is therefore visible in the returned ids.
func amortizedTestIndex(t *testing.T, dim, n int) *rpt.Index {
	t.Helper()
	idx := mustNew(t, dim,
		rpt.WithLeafCapacity(5),
		rpt.WithCandidateProjections(3),
		rpt.WithParallelThreshold(1<<30),
		rpt.WithProbeMargin(0),
		rpt.WithBruteForceFallback(false))
	rnd := rand.New(rand.NewSource(7))
	for i := 0; i < n; i++ {
		if err := idx.Add(i, makeVector(rnd, dim)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}
	return idx
}

// TestRPTIndex_AddDoesNotRebuildTree checks that a single Add on a built
// index does not rebuild the tree, and that the added id is still findable
// through the search path. The absence of a rebuild is observed through the
// seed. Two indexes are built identically under one HANN_SEED. Then each
// receives the same single Add and the same searches under a different
// HANN_SEED. A rebuild would draw the new seed and produce different trees.
// Identical results therefore prove no rebuild happened.
func TestRPTIndex_AddDoesNotRebuildTree(t *testing.T) {
	const (
		dim = 8
		n   = 5000
		k   = 5
	)
	added := []float32{9, 9, 9, 9, 9, 9, 9, 9}

	runOne := func(postBuildSeed string) [][]int {
		t.Setenv("HANN_SEED", "12345")
		idx := amortizedTestIndex(t, dim, n)
		if _, err := idx.Search(makeVector(rand.New(rand.NewSource(9)), dim), k); err != nil {
			t.Fatalf("warm-up Search failed: %v", err)
		}
		t.Setenv("HANN_SEED", postBuildSeed)
		if err := idx.Add(n, added); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
		// The added vector is far from the uniform data in [-1, 1], so an
		// exact-match query must return it first.
		got, err := idx.Search(added, 1)
		if err != nil {
			t.Fatalf("Search for the added vector failed: %v", err)
		}
		if len(got) != 1 || got[0].ID != n {
			t.Fatalf("Search for the added vector returned %v, want id %d first", got, n)
		}
		queryRnd := rand.New(rand.NewSource(11))
		results := make([][]int, 20)
		for q := range results {
			neighbors, err := idx.Search(makeVector(queryRnd, dim), k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}
			ids := make([]int, len(neighbors))
			for i, nb := range neighbors {
				ids[i] = nb.ID
			}
			results[q] = ids
		}
		return results
	}

	a := runOne("1001")
	b := runOne("2002")
	for q := range a {
		if len(a[q]) != len(b[q]) {
			t.Fatalf("query %d: result counts differ (%d vs %d), so a rebuild drew the post-build seed",
				q, len(a[q]), len(b[q]))
		}
		for i := range a[q] {
			if a[q][i] != b[q][i] {
				t.Fatalf("query %d result %d: ids differ (%d vs %d), so a rebuild drew the post-build seed",
					q, i, a[q][i], b[q][i])
			}
		}
	}
}

// TestRPTIndex_AmortizedRebuildCorrectness checks that repeated single
// Add-then-Search cycles keep the index correct, both below the rebuild
// threshold and across it. Every added id must be findable by an exact-match
// query, and Stats must report the right count throughout.
func TestRPTIndex_AmortizedRebuildCorrectness(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	const (
		dim = 8
		n   = 5000
	)
	idx := mustNew(t, dim)
	rnd := rand.New(rand.NewSource(13))
	initial := make(map[int][]float32, n)
	for i := 0; i < n; i++ {
		initial[i] = makeVector(rnd, dim)
	}
	if err := idx.BulkAdd(initial); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// A few single cycles, staying below the rebuild threshold.
	for i := 0; i < 10; i++ {
		id := n + i
		vec := makeVector(rnd, dim)
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add(%d) failed: %v", id, err)
		}
		got, err := idx.Search(vec, 1)
		if err != nil {
			t.Fatalf("Search after Add(%d) failed: %v", id, err)
		}
		if len(got) != 1 || got[0].ID != id {
			t.Fatalf("Search after Add(%d) returned %v, want the added id first", id, got)
		}
		if count := idx.Stats().Count; count != n+i+1 {
			t.Fatalf("Stats().Count = %d after Add(%d), want %d", count, id, n+i+1)
		}
	}

	// Enough further adds to cross the rebuild threshold of
	// max(64, count/4), then the same checks on a sample of the new ids.
	extra := make(map[int][]float32, n/2)
	for i := 10; i < 10+n/2; i++ {
		extra[n+i] = makeVector(rnd, dim)
	}
	if err := idx.BulkAdd(extra); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	total := n + 10 + n/2
	if count := idx.Stats().Count; count != total {
		t.Fatalf("Stats().Count = %d after BulkAdd, want %d", count, total)
	}
	// A complete search must reach every id exactly once: both the ids the
	// rebuild moved into the tree and any still in the overlay.
	all, err := idx.Search(makeVector(rnd, dim), total)
	if err != nil {
		t.Fatalf("complete Search failed: %v", err)
	}
	if len(all) != total {
		t.Fatalf("complete Search returned %d results, want %d", len(all), total)
	}
	seen := make(map[int]struct{}, total)
	for _, nb := range all {
		if nb.ID < 0 || nb.ID >= total {
			t.Fatalf("complete Search returned id %d, which was never added", nb.ID)
		}
		if _, dup := seen[nb.ID]; dup {
			t.Fatalf("complete Search returned id %d twice", nb.ID)
		}
		seen[nb.ID] = struct{}{}
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
	t.Helper()
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

// TestRPTIndex_DuplicateVectors builds a tree from many copies of the same
// vector. Every projection value is equal on such a set, so no threshold can
// separate the points. The tree build must fall back to a split by position
// instead of recursing forever, and searches must still work.
func TestRPTIndex_DuplicateVectors(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim := 8
	idx := mustNew(t, dim)
	same := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	vectors := make(map[int][]float32, 200)
	for id := 0; id < 200; id++ {
		vectors[id] = testutil.CopyVector(same)
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	neighbors, err := idx.Search(testutil.CopyVector(same), 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 5 {
		t.Fatalf("expected 5 results, got %d", len(neighbors))
	}
	for _, nb := range neighbors {
		if nb.Distance > 1e-6 {
			t.Errorf("id %d: expected distance 0 for a duplicate, got %v", nb.ID, nb.Distance)
		}
	}
}

// TestRPTIndex_MetricErrorPropagation checks that a distance failure during a
// search is returned to the caller instead of being swallowed by the
// parallel distance workers.
func TestRPTIndex_MetricErrorPropagation(t *testing.T) {
	failNow := false
	metric := core.NewMetric("rpt_flaky_test_metric", func(a, b []float32) (float64, error) {
		if failNow {
			return 0, fmt.Errorf("distance failure injected by the test")
		}
		return core.Euclidean.Distance(a, b)
	}, false)
	idx := mustNew(t, 4, rpt.WithMetric(metric))
	for id := 0; id < 10; id++ {
		if err := idx.Add(id, []float32{float32(id), 0, 0, 0}); err != nil {
			t.Fatalf("Add(%d) failed: %v", id, err)
		}
	}
	failNow = true
	if _, err := idx.Search([]float32{1, 0, 0, 0}, 3); err == nil {
		t.Error("expected error from Search with a failing metric, got none")
	}
}

// TestRPTIndex_FallbackOffShortfall runs a search that gathers fewer
// candidates than k while the brute-force fallback is disabled. The search
// must return the candidates it has, sorted, not an error or a full scan.
func TestRPTIndex_FallbackOffShortfall(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim := 8
	idx := mustNew(t, dim,
		rpt.WithLeafCapacity(5),
		rpt.WithProbeMargin(0),
		rpt.WithBruteForceFallback(false))
	data := testutil.ClusteredData(9, 100, dim, 4)
	if err := idx.BulkAdd(data); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	neighbors, err := idx.Search(testutil.CopyVector(data[0]), 50)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) == 0 || len(neighbors) >= 50 {
		t.Fatalf("expected a partial result between 1 and 49 neighbors, got %d", len(neighbors))
	}
	for i := 1; i < len(neighbors); i++ {
		if neighbors[i].Distance < neighbors[i-1].Distance {
			t.Fatalf("results are not sorted at rank %d", i)
		}
	}
	if got := idx.Stats().FallbackSearches; got != 0 {
		t.Errorf("expected no fallback searches with the fallback disabled, got %d", got)
	}
}

// TestRPTIndex_GobDecodeErrors checks that garbage bytes, a newer format
// version, and an unknown metric name are rejected by GobDecode.
func TestRPTIndex_GobDecodeErrors(t *testing.T) {
	idx := mustNew(t, 4)
	if err := idx.GobDecode([]byte("not a gob stream")); err == nil {
		t.Error("expected error decoding garbage bytes, got none")
	}

	// A file with a newer format version must be rejected. The gob decoder
	// matches struct fields by name, so a minimal struct stands in for a
	// future on-disk format.
	future := struct{ FormatVersion int }{FormatVersion: 999}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(future); err != nil {
		t.Fatalf("encoding the future format failed: %v", err)
	}
	err := idx.GobDecode(buf.Bytes())
	if err == nil {
		t.Fatal("expected error for a newer format version, got none")
	}
	if !strings.Contains(err.Error(), "format version") {
		t.Errorf("expected a format version error, got %v", err)
	}

	// An unknown metric name is an error when the decoding index has no
	// metric of its own to keep. A zero-value index has none.
	unknown := struct{ DistanceName string }{DistanceName: "no_such_metric"}
	buf.Reset()
	if err := gob.NewEncoder(&buf).Encode(unknown); err != nil {
		t.Fatalf("encoding the unknown metric failed: %v", err)
	}
	var zero rpt.Index
	err = zero.GobDecode(buf.Bytes())
	if err == nil {
		t.Fatal("expected error for an unknown metric on a zero-value index, got none")
	}
	if !strings.Contains(err.Error(), "unknown metric") {
		t.Errorf("expected an unknown metric error, got %v", err)
	}
}

// TestRPTIndex_UpdatedVectorReachableAtNewPosition checks that a point moved
// by Update or BulkUpdate is found by a query at its new position before any
// tree rebuild. The probe margin is zero and k is small, so the tree alone
// must produce the point; the brute-force fallback must stay unused.
func TestRPTIndex_UpdatedVectorReachableAtNewPosition(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim := 16
	idx := mustNew(t, dim, rpt.WithLeafCapacity(16), rpt.WithProbeMargin(0))
	rng := rand.New(rand.NewSource(1))
	pts := make(map[int][]float32, 1000)
	for id := 0; id < 1000; id++ {
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		pts[id] = vec
	}
	if err := idx.BulkAdd(pts); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	farA := make([]float32, dim)
	farB := make([]float32, dim)
	for i := range farA {
		farA[i] = 100
		farB[i] = -100
	}
	if err := idx.Update(7, farA); err != nil {
		t.Fatalf("Update failed: %v", err)
	}
	if err := idx.BulkUpdate(map[int][]float32{8: farB}); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	got, err := idx.Search(testutil.CopyVector(farA), 1)
	if err != nil {
		t.Fatalf("Search at the Update position failed: %v", err)
	}
	if len(got) != 1 || got[0].ID != 7 {
		t.Errorf("query at the updated position: got %v, want id 7", got)
	}
	got, err = idx.Search(testutil.CopyVector(farB), 1)
	if err != nil {
		t.Fatalf("Search at the BulkUpdate position failed: %v", err)
	}
	if len(got) != 1 || got[0].ID != 8 {
		t.Errorf("query at the bulk-updated position: got %v, want id 8", got)
	}
	if fb := idx.Stats().FallbackSearches; fb != 0 {
		t.Errorf("expected no fallback searches, got %d", fb)
	}
}

// TestRPTIndex_BulkUpdateInvalidLeavesIndexUnchanged checks that a batch with
// one unknown id changes nothing: every valid entry must stay at its old
// position. The check runs on many fresh indexes, because a partial
// application depends on map iteration order.
func TestRPTIndex_BulkUpdateInvalidLeavesIndexUnchanged(t *testing.T) {
	dim := 8
	for round := 0; round < 20; round++ {
		idx := mustNew(t, dim)
		for id := 0; id < 10; id++ {
			vec := make([]float32, dim)
			vec[id%dim] = float32(10 * (id + 1))
			if err := idx.Add(id, vec); err != nil {
				t.Fatalf("Add(%d) failed: %v", id, err)
			}
		}
		updates := make(map[int][]float32, 11)
		for id := 0; id < 10; id++ {
			moved := make([]float32, dim)
			moved[(id+1)%dim] = float32(-10 * (id + 1))
			updates[id] = moved
		}
		updates[99] = make([]float32, dim) // unknown id
		if err := idx.BulkUpdate(updates); err == nil {
			t.Fatal("expected error from BulkUpdate with an unknown id, got none")
		}
		for id := 0; id < 10; id++ {
			orig := make([]float32, dim)
			orig[id%dim] = float32(10 * (id + 1))
			got, err := idx.Search(orig, 1)
			if err != nil {
				t.Fatalf("round %d: Search failed: %v", round, err)
			}
			if len(got) != 1 || got[0].ID != id || got[0].Distance > 1e-6 {
				t.Fatalf("round %d: id %d moved despite the failed BulkUpdate: %v", round, id, got)
			}
		}
	}
}

// TestRPTIndex_GobDecodeSanitizesParameters loads a file whose tuning
// parameters are out of range, as a corrupt or crafted file's would be. The
// decode must fall back to the defaults instead of panicking or hanging in
// the Load-time tree build.
func TestRPTIndex_GobDecodeSanitizesParameters(t *testing.T) {
	payload := struct {
		Dimension            int
		Points               map[int][]float32
		DistanceName         string
		LeafCapacity         int
		CandidateProjections int
		ParallelThreshold    int
		ProbeMargin          float64
	}{
		Dimension:            4,
		Points:               map[int][]float32{1: {1, 0, 0, 0}, 2: {0, 1, 0, 0}},
		DistanceName:         "euclidean",
		LeafCapacity:         0,
		CandidateProjections: 0,
		ParallelThreshold:    0,
		ProbeMargin:          -1,
	}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(payload); err != nil {
		t.Fatalf("encoding the payload failed: %v", err)
	}
	var idx rpt.Index
	if err := idx.GobDecode(buf.Bytes()); err != nil {
		t.Fatalf("GobDecode failed: %v", err)
	}
	got, err := idx.Search([]float32{1, 0, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search after decode failed: %v", err)
	}
	if len(got) != 1 || got[0].ID != 1 {
		t.Errorf("expected id 1, got %v", got)
	}
	if err := idx.Add(3, []float32{0, 0, 1, 0}); err != nil {
		t.Errorf("Add after decode failed: %v", err)
	}
}

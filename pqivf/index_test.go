package pqivf_test

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"strings"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/internal/testutil"
	"github.com/habedi/hann/pqivf"
)

// newIndex constructs an index with the given parameters through New and its
// options, failing the test on a constructor error.
func newIndex(t *testing.T, dim, coarseK, numSubquantizers, pqK, kMeansIters int) *pqivf.Index {
	t.Helper()
	idx, err := pqivf.New(dim,
		pqivf.WithCoarseK(coarseK),
		pqivf.WithNumSubquantizers(numSubquantizers),
		pqivf.WithPQK(pqK),
		pqivf.WithKMeansIters(kMeansIters),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	return idx
}

func TestPQIVF_BasicOperations(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 10

	idx := newIndex(t, dim, coarseK, numSubquantizers, pqK, kMeansIters)

	// Test Add.
	vec1 := []float32{1, 2, 3, 4, 5, 6}
	if err := idx.Add(1, vec1); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	stats := idx.Stats()
	if stats.Count != 1 {
		t.Errorf("expected count 1, got %d", stats.Count)
	}

	// Test Update.
	vec1upd := []float32{6, 5, 4, 3, 2, 1}
	if err := idx.Update(1, vec1upd); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Test Delete.
	if err := idx.Delete(1); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	stats = idx.Stats()
	if stats.Count != 0 {
		t.Errorf("expected count 0 after delete, got %d", stats.Count)
	}
}

func TestPQIVF_Search(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 10
	idx := newIndex(t, dim, coarseK, numSubquantizers, pqK, kMeansIters)

	// Insert several vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {2, 2, 2, 2, 2, 2},
	}
	for id, vec := range vectors {
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add failed for id %d: %v", id, err)
		}
	}

	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	query := []float32{1, 2, 3, 4, 5, 6}
	neighbors, err := idx.Search(query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 2 {
		t.Errorf("expected 2 neighbors, got %d", len(neighbors))
	}
	// If an exact match exists, expect id 1 to be the closest.
	if neighbors[0].ID != 1 {
		t.Errorf("expected neighbor id 1 as closest, got %d", neighbors[0].ID)
	}
}

func TestPQIVF_BulkOperations(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 10
	idx := newIndex(t, dim, coarseK, numSubquantizers, pqK, kMeansIters)

	// BulkAdd a set of vectors.
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

	// BulkUpdate: update vector 2 and 3.
	updates := map[int][]float32{
		2: {1, 1, 1, 1, 1, 1},
		3: {4, 4, 4, 4, 4, 4},
	}
	if err := idx.BulkUpdate(updates); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	// Verify an update via search.
	query := []float32{1, 1, 1, 1, 1, 1}
	neighbors, err := idx.Search(query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	// Expect id 2 to be one of the closest.
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

	// BulkDelete: remove vector 1 and 4.
	if err := idx.BulkDelete([]int{1, 4}); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}
	stats = idx.Stats()
	if stats.Count != 2 {
		t.Errorf("expected count 2 after BulkDelete, got %d", stats.Count)
	}
}

func TestPQIVF_SaveLoad(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 10
	idx := newIndex(t, dim, coarseK, numSubquantizers, pqK, kMeansIters)

	// Insert a couple of vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
	}
	for id, vec := range vectors {
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add failed for id %d: %v", id, err)
		}
	}

	// Use a bytes.Buffer for in-memory serialization.
	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	newIdx := newIndex(t, dim, coarseK, numSubquantizers, pqK, kMeansIters)
	// Create a reader from the bytes of the buffer.
	r := bytes.NewReader(buf.Bytes())
	if err := newIdx.Load(r); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	stats := newIdx.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after load, got %d", len(vectors), stats.Count)
	}
}

func TestPQIVF_TrainAndSearch(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 10
	idx := newIndex(t, dim, coarseK, numSubquantizers, pqK, kMeansIters)

	// Insert data.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Search before training should fail.
	query := []float32{1, 2, 3, 4, 5, 6}
	if _, err := idx.Search(query, 1); err == nil {
		t.Fatal("expected search on untrained index to fail, but it succeeded")
	}

	// Train the index.
	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	// Search after training should succeed.
	if _, err := idx.Search(query, 1); err != nil {
		t.Fatalf("search on trained index failed: %v", err)
	}

	// BulkDelete keeps the index trained and searchable.
	if err := idx.BulkDelete([]int{1}); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}

	// Search after BulkDelete succeeds, and the deleted id is gone.
	neighbors, err := idx.Search(query, 3)
	if err != nil {
		t.Fatalf("search after BulkDelete failed: %v", err)
	}
	for _, nb := range neighbors {
		if nb.ID == 1 {
			t.Fatalf("deleted id 1 was returned by Search after BulkDelete")
		}
	}
}

// trainedIndex returns an index trained on four fixed vectors, for the tests
// that exercise mutations after training.
func trainedIndex(t *testing.T) *pqivf.Index {
	t.Helper()
	idx := newIndex(t, 6, 3, 2, 256, 10)
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {9, 9, 9, 9, 9, 9},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	return idx
}

// searchIDs returns the ids Search finds for the query, failing the test on a
// search error.
func searchIDs(t *testing.T, idx *pqivf.Index, query []float32, k int) map[int]bool {
	t.Helper()
	neighbors, err := idx.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	ids := make(map[int]bool, len(neighbors))
	for _, nb := range neighbors {
		ids[nb.ID] = true
	}
	return ids
}

func TestPQIVF_AddAfterTrain(t *testing.T) {
	idx := trainedIndex(t)

	newVec := []float32{20, 20, 20, 20, 20, 20}
	if err := idx.Add(5, newVec); err != nil {
		t.Fatalf("Add after Train failed: %v", err)
	}

	// The new id must be reachable through Search without retraining. The
	// distances are quantized, so the check is membership, not rank.
	ids := searchIDs(t, idx, newVec, 5)
	if !ids[5] {
		t.Fatalf("expected id 5 to be findable after Add on a trained index, got %v", ids)
	}
}

func TestPQIVF_BulkAddAfterTrain(t *testing.T) {
	idx := trainedIndex(t)

	if err := idx.BulkAdd(map[int][]float32{
		5: {20, 20, 20, 20, 20, 20},
		6: {-20, -20, -20, -20, -20, -20},
	}); err != nil {
		t.Fatalf("BulkAdd after Train failed: %v", err)
	}

	// Both new ids must be reachable through Search without retraining. The
	// distances are quantized, so the check is membership, not rank.
	ids := searchIDs(t, idx, []float32{20, 20, 20, 20, 20, 20}, 6)
	if !ids[5] || !ids[6] {
		t.Fatalf("expected ids 5 and 6 to be findable after BulkAdd on a trained index, got %v", ids)
	}
}

func TestPQIVF_DeleteAfterTrain(t *testing.T) {
	idx := trainedIndex(t)

	if err := idx.Delete(1); err != nil {
		t.Fatalf("Delete after Train failed: %v", err)
	}

	// The index stays searchable, the deleted id is gone, and the surviving
	// ids are still reachable.
	ids := searchIDs(t, idx, []float32{1, 2, 3, 4, 5, 6}, 3)
	if ids[1] {
		t.Fatalf("deleted id 1 was returned by Search")
	}
	if len(ids) != 3 {
		t.Fatalf("expected 3 surviving ids in the results, got %v", ids)
	}
}

func TestPQIVF_UpdateAfterTrain(t *testing.T) {
	idx := trainedIndex(t)

	newVec := []float32{30, 30, 30, 30, 30, 30}
	if err := idx.Update(2, newVec); err != nil {
		t.Fatalf("Update after Train failed: %v", err)
	}

	// The updated vector must be reflected in Search without retraining. The
	// distances are quantized, so the check is membership, not rank.
	ids := searchIDs(t, idx, newVec, 4)
	if !ids[2] {
		t.Fatalf("expected id 2 at its new vector after Update on a trained index, got %v", ids)
	}
}

func TestPQIVF_UntrainedSearchErrors(t *testing.T) {
	idx := newIndex(t, 6, 3, 2, 256, 10)
	if err := idx.Add(1, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if _, err := idx.Search([]float32{1, 2, 3, 4, 5, 6}, 1); err == nil {
		t.Fatal("expected search on a never-trained index to fail, but it succeeded")
	}
}

func TestPQIVF_EdgeCases(t *testing.T) {
	dim := 4
	coarseK := 2
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 5

	// A zero dimension is rejected by the constructor.
	if _, err := pqivf.New(0,
		pqivf.WithCoarseK(coarseK),
		pqivf.WithNumSubquantizers(numSubquantizers),
		pqivf.WithPQK(pqK),
		pqivf.WithKMeansIters(kMeansIters),
	); err == nil {
		t.Error("expected error creating a zero-dimension index, but got none")
	}

	idx := newIndex(t, dim, coarseK, numSubquantizers, pqK, kMeansIters)

	// Train on empty index.
	if err := idx.Train(); err == nil {
		t.Error("expected error training on an empty index, but got none")
	}

	// Search on empty index.
	if _, err := idx.Search([]float32{1, 2, 3, 4}, 1); err == nil {
		t.Error("expected error searching on an empty index, but got none")
	}

	// Add a vector.
	vec1 := []float32{1, 1, 1, 1}
	if err := idx.Add(1, vec1); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Add with duplicate ID.
	if err := idx.Add(1, vec1); err == nil {
		t.Error("expected error when adding a duplicate ID, but got none")
	}

	// Operations with wrong dimension vector.
	wrongDimVec := []float32{1, 2, 3}
	if err := idx.Add(2, wrongDimVec); err == nil {
		t.Error("expected error adding vector with wrong dimension, but got none")
	}
	if err := idx.Update(1, wrongDimVec); err == nil {
		t.Error("expected error updating with vector of wrong dimension, but got none")
	}
	if _, err := idx.Search(wrongDimVec, 1); err == nil {
		t.Error("expected error searching with vector of wrong dimension, but got none")
	}

	// Train with insufficient data for clustering.
	if err := idx.Train(); err == nil {
		t.Error("expected error training with insufficient data, but got none")
	}

	// Add more data to allow for training.
	if err := idx.BulkAdd(map[int][]float32{
		2: {2, 2, 2, 2},
		3: {3, 3, 3, 3},
	}); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	// Search with k=0.
	if _, err := idx.Search(vec1, 0); err == nil {
		t.Error("expected error searching with k=0, but got none")
	}

	// Search with k > number of items. The failed Update above left id 1
	// in place, so the index holds three entries.
	neighbors, err := idx.Search(vec1, 5)
	if err != nil {
		t.Fatalf("Search with k > num items failed: %v", err)
	}
	if len(neighbors) != 3 {
		t.Errorf("expected 3 neighbors, got %d", len(neighbors))
	}
}

func TestPQIVF_UpdateFailureKeepsEntry(t *testing.T) {
	dim := 4
	idx := newIndex(t, dim, 2, 2, 256, 5)

	if err := idx.Add(1, []float32{1, 1, 1, 1}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// An update with a vector of the wrong dimension must fail and leave
	// the original entry in place.
	if err := idx.Update(1, []float32{1, 2, 3}); err == nil {
		t.Fatal("expected error updating with a vector of the wrong dimension, but got none")
	}
	if stats := idx.Stats(); stats.Count != 1 {
		t.Fatalf("expected count 1 after failed Update, got %d", stats.Count)
	}

	// The entry must still be updatable with a valid vector.
	if err := idx.Update(1, []float32{2, 2, 2, 2}); err != nil {
		t.Fatalf("Update with a valid vector failed: %v", err)
	}
}

func TestPQIVF_BulkUpdateFailureKeepsEntries(t *testing.T) {
	dim := 4
	idx := newIndex(t, dim, 2, 2, 256, 5)

	vectors := map[int][]float32{
		1: {1, 1, 1, 1},
		2: {2, 2, 2, 2},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	updates := map[int][]float32{
		1: {3, 3, 3, 3},
		2: {1, 2, 3},
	}
	if err := idx.BulkUpdate(updates); err == nil {
		t.Fatal("expected error from BulkUpdate with a vector of the wrong dimension, but got none")
	}
	if stats := idx.Stats(); stats.Count != 2 {
		t.Fatalf("expected count 2 after failed BulkUpdate, got %d", stats.Count)
	}
}

func TestPQIVF_InvalidCoarseK(t *testing.T) {
	for _, coarseK := range []int{0, -1} {
		if _, err := pqivf.New(4, pqivf.WithCoarseK(coarseK)); err == nil {
			t.Errorf("expected error for coarseK=%d, but got none", coarseK)
		}
	}
}

// TestPQIVF_NewValidation covers the remaining constructor errors: a
// non-positive dimension, PQ codebook size, k-means iteration count, or
// subquantizer count, and a dimension that the subquantizer count does not
// divide.
func TestPQIVF_NewValidation(t *testing.T) {
	cases := []struct {
		name string
		dim  int
		opts []pqivf.Option
	}{
		{name: "zero dimension", dim: 0},
		{name: "negative dimension", dim: -4},
		{name: "zero pqK", dim: 4, opts: []pqivf.Option{pqivf.WithPQK(0)}},
		{name: "negative pqK", dim: 4, opts: []pqivf.Option{pqivf.WithPQK(-1)}},
		{name: "zero kMeansIters", dim: 4, opts: []pqivf.Option{pqivf.WithKMeansIters(0)}},
		{name: "negative kMeansIters", dim: 4, opts: []pqivf.Option{pqivf.WithKMeansIters(-1)}},
		{name: "zero numSubquantizers", dim: 4, opts: []pqivf.Option{pqivf.WithNumSubquantizers(0)}},
		{name: "indivisible dimension", dim: 6, opts: []pqivf.Option{pqivf.WithNumSubquantizers(4)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := pqivf.New(tc.dim, tc.opts...); err == nil {
				t.Errorf("expected error, but got none")
			}
		})
	}
}

// TestPQIVF_DefaultNumSubquantizers checks the default subquantizer rule: the
// largest of 8, 4, 2, and 1 that divides the dimension is chosen, so every
// positive dimension yields a valid index without options.
func TestPQIVF_DefaultNumSubquantizers(t *testing.T) {
	for _, dim := range []int{16, 12, 6, 7} {
		if _, err := pqivf.New(dim); err != nil {
			t.Errorf("New(%d) with default options failed: %v", dim, err)
		}
	}
}

// clusteredVectors returns three vectors around each of four well-separated
// centers, keyed by ids 0 through 11, so coarse clustering with four clusters
// cannot put every vector into one cluster.
func clusteredVectors() map[int][]float32 {
	vectors := make(map[int][]float32, 12)
	id := 0
	for _, base := range []float32{0, 100, 200, 300} {
		for j := 0; j < 3; j++ {
			off := base + float32(j)
			vectors[id] = []float32{off, off, off, off}
			id++
		}
	}
	return vectors
}

// TestPQIVF_CandidateClustersAndFallbackOff checks the observable effect of
// WithCandidateClusters and WithBruteForceFallback: with one probed cluster
// and the fallback off, a k larger than one cluster's entries returns fewer
// than k results, while the same search with the fallback on returns all of
// them through the brute-force scan.
func TestPQIVF_CandidateClustersAndFallbackOff(t *testing.T) {
	vectors := clusteredVectors()
	query := vectors[0]
	k := len(vectors)

	restricted, err := pqivf.New(4,
		pqivf.WithCoarseK(4),
		pqivf.WithNumSubquantizers(2),
		pqivf.WithPQK(4),
		pqivf.WithKMeansIters(10),
		pqivf.WithCandidateClusters(1),
		pqivf.WithBruteForceFallback(false),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if err := restricted.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := restricted.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	got, err := restricted.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(got) == 0 || len(got) >= k {
		t.Errorf("expected between 1 and %d results from a single probed cluster, got %d", k-1, len(got))
	}
	found := false
	for _, n := range got {
		if n.ID == 0 {
			found = true
		}
	}
	if !found {
		t.Errorf("expected the exact match id 0 in the probed cluster, got %v", got)
	}
	if fallbacks := restricted.Stats().FallbackSearches; fallbacks != 0 {
		t.Errorf("expected 0 fallback searches with the fallback off, got %d", fallbacks)
	}

	// The same configuration with the fallback on must fill the shortfall
	// with a brute-force scan.
	fallback, err := pqivf.New(4,
		pqivf.WithCoarseK(4),
		pqivf.WithNumSubquantizers(2),
		pqivf.WithPQK(4),
		pqivf.WithKMeansIters(10),
		pqivf.WithCandidateClusters(1),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if err := fallback.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := fallback.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	got, err = fallback.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(got) != k {
		t.Errorf("expected %d results with the fallback on, got %d", k, len(got))
	}
	if fallbacks := fallback.Stats().FallbackSearches; fallbacks < 1 {
		t.Errorf("expected at least 1 fallback search, got %d", fallbacks)
	}
}

func TestPQIVF_BulkAddValidation(t *testing.T) {
	idx := newIndex(t, 4, 2, 2, 4, 5)
	if err := idx.Add(1, []float32{1, 1, 1, 1}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// A row with the wrong dimension must fail.
	if err := idx.BulkAdd(map[int][]float32{2: {1, 2}}); err == nil {
		t.Error("expected error for a bulk row with the wrong dimension, got none")
	}

	// An id that is already pending must fail.
	if err := idx.BulkAdd(map[int][]float32{1: {2, 2, 2, 2}}); err == nil {
		t.Error("expected error for an id already in pending vectors, got none")
	}
	if got := idx.Stats().Count; got != 1 {
		t.Errorf("expected count 1 after failed BulkAdd calls, got %d", got)
	}

	// An id that is already clustered must fail too.
	if err := idx.BulkAdd(map[int][]float32{2: {5, 5, 5, 5}}); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	if err := idx.BulkAdd(map[int][]float32{1: {2, 2, 2, 2}}); err == nil {
		t.Error("expected error for an id already in a cluster, got none")
	}
	if got := idx.Stats().Count; got != 2 {
		t.Errorf("expected count 2 after the failed BulkAdd on a trained index, got %d", got)
	}
}

// TestPQIVF_BulkDeleteUnknownID checks that an unknown id is skipped and the
// known ids are still removed.
func TestPQIVF_BulkDeleteUnknownID(t *testing.T) {
	idx := newIndex(t, 4, 2, 2, 4, 5)
	if err := idx.BulkAdd(map[int][]float32{
		1: {1, 1, 1, 1},
		2: {2, 2, 2, 2},
	}); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := idx.BulkDelete([]int{99, 2}); err != nil {
		t.Fatalf("BulkDelete with an unknown id failed: %v", err)
	}
	if got := idx.Stats().Count; got != 1 {
		t.Fatalf("expected count 1 after BulkDelete, got %d", got)
	}
}

// TestPQIVF_BulkUpdateUnknownID checks that updating an id that is not in the
// index fails with an error.
func TestPQIVF_BulkUpdateUnknownID(t *testing.T) {
	idx := newIndex(t, 4, 2, 2, 4, 5)
	if err := idx.Add(1, []float32{1, 1, 1, 1}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if err := idx.BulkUpdate(map[int][]float32{99: {2, 2, 2, 2}}); err == nil {
		t.Error("expected error updating an unknown id, got none")
	}
	if got := idx.Stats().Count; got != 1 {
		t.Errorf("expected count 1 after the failed BulkUpdate, got %d", got)
	}
}

// TestPQIVF_ZeroValueIndexRejectsAdd checks that an index that was not built
// by New reports a clear error instead of accepting data.
func TestPQIVF_ZeroValueIndexRejectsAdd(t *testing.T) {
	var idx pqivf.Index
	err := idx.Add(1, []float32{1, 2, 3, 4})
	if err == nil {
		t.Fatal("expected error adding to a zero-value index, got none")
	}
	if !strings.Contains(err.Error(), "zero-dimension") {
		t.Errorf("expected a zero-dimension error, got %v", err)
	}
}

// TestPQIVF_SearchWrongDimensionAfterTrain checks the query dimension check on
// a trained index, where the untrained-index error cannot mask it.
func TestPQIVF_SearchWrongDimensionAfterTrain(t *testing.T) {
	idx := newIndex(t, 4, 2, 2, 4, 5)
	if err := idx.BulkAdd(map[int][]float32{
		1: {1, 1, 1, 1},
		2: {2, 2, 2, 2},
	}); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	if _, err := idx.Search([]float32{1, 2}, 1); err == nil {
		t.Error("expected error for a query with the wrong dimension, got none")
	}
}

func TestPQIVF_GobDecodeErrors(t *testing.T) {
	idx := newIndex(t, 4, 2, 2, 4, 5)

	// Bytes that are not a gob stream must fail.
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
}

// TestPQIVF_GobDecodeLegacyPendingVectors checks that a file written before
// the PendingVectors field existed loads into a usable index: the decoded nil
// map is replaced, so a later Add works. The gob decoder matches struct fields
// by name, so a struct without the field stands in for the old format.
func TestPQIVF_GobDecodeLegacyPendingVectors(t *testing.T) {
	legacy := struct {
		Dimension        int
		CoarseK          int
		NumSubquantizers int
		PqK              int
		KMeansIters      int
	}{Dimension: 4, CoarseK: 2, NumSubquantizers: 2, PqK: 4, KMeansIters: 5}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(legacy); err != nil {
		t.Fatalf("encoding the legacy index failed: %v", err)
	}
	idx := &pqivf.Index{}
	if err := idx.GobDecode(buf.Bytes()); err != nil {
		t.Fatalf("GobDecode of the legacy index failed: %v", err)
	}
	if err := idx.Add(1, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Add failed after decoding a legacy index: %v", err)
	}
	if got := idx.Stats().Count; got != 1 {
		t.Errorf("expected count 1 after Add, got %d", got)
	}
}

// pqivfFactory returns the factory shared by the property-based and the
// concurrency tests.
func pqivfFactory(t *testing.T) testutil.Factory {
	return testutil.Factory{
		New: func() core.Index {
			return newIndex(t, 16, 2, 2, 4, 5)
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
}

// TestPQIVF_FallbackSearchCounter checks that a search that falls back to a
// brute-force scan is visible in Stats.
func TestPQIVF_FallbackSearchCounter(t *testing.T) {
	// More clusters than the search probes, so a large k cannot be satisfied
	// from the probed clusters alone and the brute-force fallback triggers.
	pq := newIndex(t, 4, 8, 2, 2, 5)
	for id := 0; id < 40; id++ {
		vec := []float32{float32(id), float32(id + 1), float32(id + 2), float32(id + 3)}
		if err := pq.Add(id, vec); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}
	if err := pq.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	if got := pq.Stats().FallbackSearches; got != 0 {
		t.Fatalf("expected 0 fallback searches before any search, got %d", got)
	}
	if _, err := pq.Search([]float32{0, 1, 2, 3}, 40); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if got := pq.Stats().FallbackSearches; got < 1 {
		t.Fatalf("expected at least 1 fallback search, got %d", got)
	}
}

// TestPQIVF_DeleteAllThenSearch empties a trained index through Delete and
// checks that a search returns no results and no error, and that a
// subsequent Add makes the index searchable again.
func TestPQIVF_DeleteAllThenSearch(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim := 4
	idx := newIndex(t, dim, 2, 2, 2, 5)
	vectors := map[int][]float32{
		1: {1, 0, 0, 0},
		2: {0, 1, 0, 0},
		3: {0, 0, 1, 0},
		4: {0, 0, 0, 1},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	for id := range vectors {
		if err := idx.Delete(id); err != nil {
			t.Fatalf("Delete(%d) failed: %v", id, err)
		}
	}
	if got := idx.Stats().Count; got != 0 {
		t.Fatalf("expected count 0 after deleting everything, got %d", got)
	}
	neighbors, err := idx.Search([]float32{1, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("Search on the emptied index failed: %v", err)
	}
	if len(neighbors) != 0 {
		t.Fatalf("expected no results from the emptied index, got %v", neighbors)
	}
	if err := idx.Add(5, []float32{1, 1, 0, 0}); err != nil {
		t.Fatalf("Add after emptying failed: %v", err)
	}
	neighbors, err = idx.Search([]float32{1, 1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search after the re-add failed: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != 5 {
		t.Errorf("expected id 5 after the re-add, got %v", neighbors)
	}
}

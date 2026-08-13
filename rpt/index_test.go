package rpt_test

import (
	"bytes"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/rpt"
)

const (
	defaultLeafCapacity         = 10
	defaultCandidateProjections = 3
	defaultParallelThreshold    = 100
	defaultProbeMargin          = 0.15
)

func TestRPTIndex_BasicOperations(t *testing.T) {
	dim := 6
	idx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)

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
	idx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)

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
	idx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)

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
	idx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)
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
	newIdx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)
	r := bytes.NewReader(buf.Bytes())
	if err := newIdx.Load(r); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	stats := newIdx.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after load, got %d", len(vectors), stats.Count)
	}
}

func TestRPTIndex_ConcurrentOperations(t *testing.T) {
	dim := 6
	idx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)
	numVectors := 1000
	var wg sync.WaitGroup

	for i := 0; i < numVectors; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			vec := []float32{
				float32(id),
				float32(id + 1),
				float32(id + 2),
				float32(id + 3),
				float32(id + 4),
				float32(id + 5),
			}
			if err := idx.Add(id, vec); err != nil {
				t.Errorf("Add failed for id %d: %v", id, err)
			}
		}(i)
	}
	wg.Wait()

	stats := idx.Stats()
	if stats.Count != numVectors {
		t.Errorf("expected %d vectors, got %d", numVectors, stats.Count)
	}
}

func TestRPTIndex_ErrorOnWrongVectorDimension(t *testing.T) {
	dim := 6
	idx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)

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
	idx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)

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
	emptyIdx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)
	var buf bytes.Buffer
	if err := emptyIdx.Save(&buf); err != nil {
		t.Fatalf("Save on empty index failed: %v", err)
	}
	newEmptyIdx := rpt.NewRPTIndex(dim, defaultLeafCapacity, defaultCandidateProjections,
		defaultParallelThreshold, defaultProbeMargin)
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

// TestRPTIndex_ConcurrentSearchWithWrites checks that Search does not read the
// point map without holding the lock while other goroutines mutate the index.
// Run with the race detector to catch unlocked map access.
func TestRPTIndex_ConcurrentSearchWithWrites(t *testing.T) {
	dim := 8
	idx := rpt.NewRPTIndex(dim, 5, 2, 1<<30, 0.1)
	rnd := rand.New(rand.NewSource(1))
	for i := 0; i < 500; i++ {
		if err := idx.Add(i, makeVector(rnd, dim)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}
	query := makeVector(rnd, dim)
	if _, err := idx.Search(query, 5); err != nil {
		t.Fatalf("warm-up Search failed: %v", err)
	}

	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				if _, err := idx.Search(query, 5); err != nil {
					t.Errorf("Search failed during concurrent writes: %v", err)
					return
				}
			}
		}()
	}
	for w := 0; w < 2; w++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			localRnd := rand.New(rand.NewSource(int64(offset)))
			for i := 0; i < 100; i++ {
				id := offset*250 + i
				if err := idx.Delete(id); err != nil {
					t.Errorf("Delete failed: %v", err)
					return
				}
				if err := idx.Add(id, makeVector(localRnd, dim)); err != nil {
					t.Errorf("Add failed: %v", err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
}

// TestRPTIndex_SaveWithConcurrentWriters checks that Save does not deadlock
// when writers contend for the lock while a save is in progress.
func TestRPTIndex_SaveWithConcurrentWriters(t *testing.T) {
	dim := 8
	idx := rpt.NewRPTIndex(dim, 5, 2, 1<<30, 0.1)
	rnd := rand.New(rand.NewSource(2))
	for i := 0; i < 100; i++ {
		if err := idx.Add(i, makeVector(rnd, dim)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		var wg sync.WaitGroup
		for w := 0; w < 4; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := 0; i < 200; i++ {
					var buf bytes.Buffer
					if err := idx.Save(&buf); err != nil {
						t.Errorf("Save failed: %v", err)
						return
					}
				}
			}()
		}
		for w := 0; w < 4; w++ {
			wg.Add(1)
			go func(offset int) {
				defer wg.Done()
				localRnd := rand.New(rand.NewSource(int64(offset)))
				for i := 0; i < 200; i++ {
					id := 10000 + offset*1000 + i
					if err := idx.Add(id, makeVector(localRnd, dim)); err != nil {
						t.Errorf("Add failed: %v", err)
						return
					}
					if err := idx.Delete(id); err != nil {
						t.Errorf("Delete failed: %v", err)
						return
					}
				}
			}(w)
		}
		wg.Wait()
	}()

	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("Save deadlocked with concurrent writers")
	}
}

// TestRPTIndex_ConcurrentMultiProbeSearch checks that two concurrent searches
// cannot write into the same backing array when the multi-probe branch merges
// candidate lists from leaves. Run with the race detector.
func TestRPTIndex_ConcurrentMultiProbeSearch(t *testing.T) {
	dim := 8
	// A very large probe margin makes every internal node probe both children,
	// and a leaf capacity of 5 allows leaves whose slices have spare capacity
	// that can hold the sibling's points.
	idx := rpt.NewRPTIndex(dim, 5, 1, 1<<30, 1e9)
	rnd := rand.New(rand.NewSource(3))
	for i := 0; i < 500; i++ {
		if err := idx.Add(i, makeVector(rnd, dim)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Rebuild the tree several times so that different random trees are searched.
	for round := 0; round < 10; round++ {
		// Mark the tree dirty so the next search rebuilds it.
		if err := idx.Delete(0); err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		if err := idx.Add(0, makeVector(rnd, dim)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
		query := makeVector(rnd, dim)
		if _, err := idx.Search(query, 5); err != nil {
			t.Fatalf("warm-up Search failed: %v", err)
		}

		var wg sync.WaitGroup
		for w := 0; w < 8; w++ {
			wg.Add(1)
			go func(seed int64) {
				defer wg.Done()
				localRnd := rand.New(rand.NewSource(seed))
				for i := 0; i < 20; i++ {
					q := makeVector(localRnd, dim)
					if _, err := idx.Search(q, 5); err != nil {
						t.Errorf("Search failed: %v", err)
						return
					}
				}
			}(int64(w))
		}
		wg.Wait()
	}
}

// TestRPTIndex_SeedReproducibility checks that two indexes built from the same
// data with the same HANN_SEED return the same search results.
func TestRPTIndex_SeedReproducibility(t *testing.T) {
	t.Setenv("HANN_SEED", "12345")
	dim := 8
	build := func() *rpt.RPTIndex {
		idx := rpt.NewRPTIndex(dim, 5, 3, 1<<30, 0)
		idx.AllowBruteForceFallback = false
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

// TestRPTIndex_SaveLoadDistance checks that the configured distance name is
// reported by Stats, survives a save and load round-trip, and that the
// distance function is restored when loading into a zero-value index.
func TestRPTIndex_SaveLoadDistance(t *testing.T) {
	dim := 4
	idx := rpt.NewRPTIndex(dim, 4, 2, 100, 0.1)
	idx.Distance = core.Distances["manhattan"]
	idx.DistanceName = "manhattan"
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

	loaded := &rpt.RPTIndex{}
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

// TestRPTIndex_ConstructorValidation checks that the constructor rejects
// parameter values that would break the index.
func TestRPTIndex_ConstructorValidation(t *testing.T) {
	t.Run("non-positive leaf capacity", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("expected panic for leafCapacity <= 0, but got none")
			}
		}()
		rpt.NewRPTIndex(4, 0, 2, 100, 0.1)
	})
	t.Run("non-positive candidate projections", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("expected panic for candidateProjections <= 0, but got none")
			}
		}()
		rpt.NewRPTIndex(4, 4, 0, 100, 0.1)
	})
}

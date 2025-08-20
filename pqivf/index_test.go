package pqivf_test

import (
	"bytes"
	"sync"
	"testing"

	"github.com/habedi/hann/pqivf"
)

func TestPQIVF_BasicOperations(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 10

	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers, pqK, kMeansIters)

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
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers, pqK, kMeansIters)

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
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers, pqK, kMeansIters)

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
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers, pqK, kMeansIters)

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

	newIdx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers, pqK, kMeansIters)
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
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers, pqK, kMeansIters)

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

	// BulkDelete should invalidate training.
	if err := idx.BulkDelete([]int{1}); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}

	// Search after BulkDelete should fail.
	if _, err := idx.Search(query, 1); err == nil {
		t.Fatal("expected search after BulkDelete to fail, but it succeeded")
	}
}

func TestPQIVF_ConcurrentOperations(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 10
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers, pqK, kMeansIters)
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

func TestPQIVF_EdgeCases(t *testing.T) {
	dim := 4
	coarseK := 2
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 5

	// Test with zero dimension.
	zeroDimIdx := pqivf.NewPQIVFIndex(0, coarseK, numSubquantizers, pqK, kMeansIters)
	if err := zeroDimIdx.Add(1, []float32{}); err == nil {
		t.Error("expected error when adding a vector to a zero-dimension index, but got none")
	}

	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers, pqK, kMeansIters)

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

	// Search with k > number of items.
	neighbors, err := idx.Search(vec1, 5)
	if err != nil {
		t.Fatalf("Search with k > num items failed: %v", err)
	}
	if len(neighbors) != 2 {
		t.Errorf("expected 2 neighbors, got %d", len(neighbors))
	}
}

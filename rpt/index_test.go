package rpt_test

import (
	"os"
	"sync"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/rpt"
)

func TestRPTIndex_BasicOperations(t *testing.T) {
	dim := 6
	distanceName := "euclidean"
	// Pass core.Euclidean and "euclidean" as the distance function and name.
	idx := rpt.NewRPTIndex(dim, core.Distances[distanceName], distanceName)

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

func TestRPTIndex_Search(t *testing.T) {
	dim := 6
	idx := rpt.NewRPTIndex(dim, core.Euclidean, "euclidean")

	// Insert several vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {2, 2, 2, 2, 2, 2},
		5: {1, 2, 2, 3, 4, 5},
	}
	// Use BulkAdd for convenience.
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
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
	idx := rpt.NewRPTIndex(dim, core.Euclidean, "euclidean")

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

	// Verify an update via search.
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
	idx := rpt.NewRPTIndex(dim, core.Euclidean, "euclidean")
	// Insert a couple of vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
	}
	if err := idx.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	filePath := "test_rpt.gob"
	defer os.Remove(filePath)

	if err := idx.Save(filePath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	newIdx := rpt.NewRPTIndex(dim, core.Euclidean, "euclidean")
	if err := newIdx.Load(filePath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	stats := newIdx.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after load, got %d", len(vectors), stats.Count)
	}
}

func TestRPTIndex_ConcurrentOperations(t *testing.T) {
	dim := 6
	idx := rpt.NewRPTIndex(dim, core.Euclidean, "euclidean")
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

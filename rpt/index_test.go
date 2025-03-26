package rpt_test

import (
	"bytes"
	"sync"
	"testing"

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

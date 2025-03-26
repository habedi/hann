package hnsw_test

import (
	"os"
	"sync"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/hnsw"
)

func TestHNSWIndex_AddAndStats(t *testing.T) {
	dim := 6
	distanceName := "euclidean"
	index := hnsw.NewHNSW(dim, 5, 10, core.Distances[distanceName], distanceName)

	// Test single Add.
	if err := index.Add(1, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Test dimension mismatch.
	err := index.Add(2, []float32{1, 2, 3})
	if err == nil {
		t.Fatal("expected error due to dimension mismatch, got none")
	}

	// Test duplicate id.
	err = index.Add(1, []float32{6, 5, 4, 3, 2, 1})
	if err == nil {
		t.Fatal("expected error due to duplicate id, got none")
	}

	// Verify stats.
	stats := index.Stats()
	if stats.Count != 1 {
		t.Errorf("expected count 1 after one Add, got %d", stats.Count)
	}
}

func TestHNSWIndex_Delete(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")

	// Arrange: add two vectors.
	if err := index.Add(1, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if err := index.Add(2, []float32{6, 5, 4, 3, 2, 1}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Act: delete id 1.
	if err := index.Delete(1); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// Assert: stats count should be 1.
	stats := index.Stats()
	if stats.Count != 1 {
		t.Errorf("expected count 1 after Delete, got %d", stats.Count)
	}

	// Delete non-existent id.
	if err := index.Delete(10); err == nil {
		t.Error("expected error when deleting non-existent id, got none")
	}
}

func TestHNSWIndex_Update(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")

	// Arrange: add a vector.
	if err := index.Add(1, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Act: update with new vector.
	if err := index.Update(1, []float32{6, 6, 6, 6, 6, 6}); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Assert: search with updated vector.
	query := []float32{6, 6, 6, 6, 6, 6}
	neighbors, err := index.Search(query, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) == 0 || neighbors[0].ID != 1 {
		t.Errorf("expected id 1 as nearest neighbor after Update, got %v", neighbors)
	}

	// Update non-existent id.
	err = index.Update(10, []float32{1, 1, 1, 1, 1, 1})
	if err == nil {
		t.Error("expected error when updating non-existent id, got none")
	}

	// Update with wrong dimension.
	err = index.Update(1, []float32{1, 2, 3})
	if err == nil {
		t.Error("expected error due to dimension mismatch in update, got none")
	}
}

func TestHNSWIndex_BulkAdd(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")

	// Arrange: Create a set of 5 vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {2, 2, 2, 2, 2, 2},
		5: {3, 3, 3, 3, 3, 3},
	}

	// Act: Bulk add the vectors.
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Assert: Check the index count.
	stats := index.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after BulkAdd, got %d", len(vectors), stats.Count)
	}
}

func TestHNSWIndex_BulkDelete(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")

	// Arrange: Bulk add a set of vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {2, 2, 2, 2, 2, 2},
		5: {3, 3, 3, 3, 3, 3},
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Act: Bulk delete some ids.
	deleteIDs := []int{2, 4}
	if err := index.BulkDelete(deleteIDs); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}

	// Assert: Verify the count and ensure the deleted ids are gone.
	stats := index.Stats()
	expectedCount := len(vectors) - len(deleteIDs)
	if stats.Count != expectedCount {
		t.Errorf("expected count %d after BulkDelete, got %d", expectedCount, stats.Count)
	}

	// Optionally, perform a search to check that deleted vectors are not returned.
	query := []float32{6, 5, 4, 3, 2, 1}
	neighbors, err := index.Search(query, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	for _, n := range neighbors {
		if n.ID == 2 || n.ID == 4 {
			t.Errorf("deleted id %d returned in search results", n.ID)
		}
	}
}

func TestHNSWIndex_BulkUpdate(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")

	// Arrange: Bulk add a set of vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Act: Bulk update vectors for some ids.
	updates := map[int][]float32{
		1: {6, 6, 6, 6, 6, 6},
		3: {2, 2, 2, 2, 2, 2},
	}
	if err := index.BulkUpdate(updates); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	// Assert: For example, search with a query similar to the updated vector of id 1.
	query := []float32{6, 6, 6, 6, 6, 6}
	neighbors, err := index.Search(query, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) == 0 || neighbors[0].ID != 1 {
		t.Errorf("expected id 1 as nearest neighbor after BulkUpdate, got %v", neighbors)
	}
}

func TestHNSWIndex_SaveLoad(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")

	// Arrange: add some vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Create a temporary file for saving.
	tmpFile, err := os.CreateTemp("", "temp_index_*.gob")
	if err != nil {
		t.Fatalf("failed to create temporary file: %v", err)
	}
	tmpPath := tmpFile.Name()
	// Save the index using the io.Writer.
	if err := index.Save(tmpFile); err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	tmpFile.Close()
	defer os.Remove(tmpPath)

	// Open the file for reading.
	readFile, err := os.Open(tmpPath)
	if err != nil {
		t.Fatalf("failed to open temporary file: %v", err)
	}
	defer readFile.Close()

	// Create a new index and load the saved state using the io.Reader.
	newIndex := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	if err := newIndex.Load(readFile); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Assert: check that stats match.
	stats := newIndex.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after Load, got %d", len(vectors), stats.Count)
	}
}

func TestHNSWIndex_ConcurrentBulkOperations(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	numVectors := 1000

	// Arrange: prepare a map of vectors.
	vectors := make(map[int][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = []float32{
			float32(i),
			float32(i + 1),
			float32(i + 2),
			float32(i + 3),
			float32(i + 4),
			float32(i + 5),
		}
	}

	// Act: perform BulkAdd.
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Prepare updates: update half the vectors.
	updates := make(map[int][]float32)
	for i := 0; i < numVectors; i += 2 {
		updates[i] = []float32{
			float32(i + 10),
			float32(i + 11),
			float32(i + 12),
			float32(i + 13),
			float32(i + 14),
			float32(i + 15),
		}
	}

	// Prepare deletions: delete one-quarter of the vectors.
	var deleteIDs []int
	for i := 0; i < numVectors; i += 4 {
		deleteIDs = append(deleteIDs, i)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		if err := index.BulkUpdate(updates); err != nil {
			t.Errorf("BulkUpdate failed: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := index.BulkDelete(deleteIDs); err != nil {
			t.Errorf("BulkDelete failed: %v", err)
		}
	}()
	wg.Wait()

	// Assert: final count.
	expected := numVectors - len(deleteIDs)
	stats := index.Stats()
	if stats.Count != expected {
		t.Errorf("expected count %d after concurrent bulk operations, got %d", expected,
			stats.Count)
	}
}

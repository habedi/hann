package hnsw_test

import (
	"bytes"
	"io"
	"os"
	"sync"
	"testing"
	"time"

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

// testVector returns a 6-dimensional vector derived from i, spaced far
// apart so that nearest-neighbor results are unambiguous.
func testVector(i int) []float32 {
	base := float32(i * 100)
	return []float32{base, base + 1, base + 2, base + 3, base + 4, base + 5}
}

func TestHNSWIndex_SaveConcurrentWithAdd(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	for i := 0; i < 100; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				if err := index.Save(io.Discard); err != nil {
					t.Errorf("Save failed: %v", err)
					return
				}
			}
		}()
		go func() {
			defer wg.Done()
			for i := 1000; i < 1200; i++ {
				if err := index.Add(i, testVector(i)); err != nil {
					t.Errorf("Add failed: %v", err)
					return
				}
			}
		}()
		wg.Wait()
	}()

	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("Save deadlocked while running concurrently with Add")
	}
}

func TestHNSWIndex_SaveLoadEntryPointZero(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	if err := index.Add(0, testVector(0)); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	var buf bytes.Buffer
	if err := index.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	neighbors, err := loaded.Search(testVector(0), 1)
	if err != nil {
		t.Fatalf("Search failed after loading an index whose entry point has id 0: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != 0 {
		t.Errorf("expected id 0 as nearest neighbor after Load, got %v", neighbors)
	}
}

func TestHNSWIndex_LoadIntoZeroValueIndex(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	for i := 1; i <= 10; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	var buf bytes.Buffer
	if err := index.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load into a zero-value index: the distance function and the internal
	// maps must be restored from the serialized state.
	loaded := &hnsw.HNSWIndex{}
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	neighbors, err := loaded.Search(testVector(3), 1)
	if err != nil {
		t.Fatalf("Search failed after Load into a zero-value index: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != 3 {
		t.Errorf("expected id 3 as nearest neighbor after Load, got %v", neighbors)
	}

	// Adding after Load must not panic on internal maps.
	if err := loaded.Add(11, testVector(11)); err != nil {
		t.Fatalf("Add failed after Load into a zero-value index: %v", err)
	}
}

func TestHNSWIndex_LoadCustomDistance(t *testing.T) {
	dim := 6
	custom := func(a, b []float32) (float64, error) {
		return core.Euclidean(a, b)
	}
	index := hnsw.NewHNSW(dim, 5, 10, custom, "custom_metric")
	for i := 1; i <= 5; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	var buf bytes.Buffer
	if err := index.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Loading into a zero-value index must fail, because the distance name
	// is unknown and no distance function is present.
	saved := buf.Bytes()
	empty := &hnsw.HNSWIndex{}
	if err := empty.Load(bytes.NewReader(saved)); err == nil {
		t.Error("expected error when loading an unknown distance into a zero-value index, got none")
	}

	// Loading onto an index constructed with the custom function must keep it.
	target := hnsw.NewHNSW(dim, 5, 10, custom, "custom_metric")
	if err := target.Load(bytes.NewReader(saved)); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	neighbors, err := target.Search(testVector(2), 1)
	if err != nil {
		t.Fatalf("Search failed after Load with a custom distance: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != 2 {
		t.Errorf("expected id 2 as nearest neighbor after Load, got %v", neighbors)
	}
}

func TestHNSWIndex_DeleteEntryPointAfterLoad(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	for i := 1; i <= 10; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	var buf bytes.Buffer
	if err := index.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	loaded := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Delete all ids but one; after each deletion the survivors must still
	// be reachable, which requires the level bookkeeping to have been
	// rebuilt by Load.
	for i := 1; i <= 9; i++ {
		if err := loaded.Delete(i); err != nil {
			t.Fatalf("Delete(%d) failed: %v", i, err)
		}
		neighbors, err := loaded.Search(testVector(10), 1)
		if err != nil {
			t.Fatalf("Search failed after Delete(%d) on a loaded index: %v", i, err)
		}
		if len(neighbors) != 1 || neighbors[0].ID != 10 {
			t.Errorf("expected id 10 as nearest neighbor after Delete(%d), got %v", i, neighbors)
		}
	}
}

func TestHNSWIndex_DeleteEntryPointAfterBulkAdd(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	vectors := make(map[int][]float32, 50)
	for i := 1; i <= 50; i++ {
		vectors[i] = testVector(i)
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Repeatedly delete the current entry point; the survivors must remain
	// reachable, which requires BulkAdd to have maintained the level
	// bookkeeping that Delete uses to pick a new entry point.
	for count := 50; count > 1; count-- {
		epID := index.EntryPoint.ID
		if err := index.Delete(epID); err != nil {
			t.Fatalf("Delete(%d) failed: %v", epID, err)
		}
		if _, err := index.Search(testVector(1), 1); err != nil {
			t.Fatalf("Search failed with %d nodes left after deleting entry point %d: %v",
				count-1, epID, err)
		}
	}
}

func TestHNSWIndex_DeleteAfterBulkDelete(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	for i := 1; i <= 100; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Bulk delete every node above level 0, so any stale level bookkeeping
	// left behind by BulkDelete points only at removed nodes.
	var deleteIDs []int
	for id, node := range index.Nodes {
		if node.Level >= 1 {
			deleteIDs = append(deleteIDs, id)
		}
	}
	if len(deleteIDs) == 0 {
		t.Skip("no nodes above level 0; cannot exercise the entry point recovery")
	}
	if err := index.BulkDelete(deleteIDs); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}
	survivors := 100 - len(deleteIDs)
	if survivors < 2 {
		t.Skip("fewer than two survivors; cannot exercise the entry point recovery")
	}

	// Deleting the entry point must pick a surviving node as the new entry
	// point, not a node that BulkDelete already removed.
	epID := index.EntryPoint.ID
	if err := index.Delete(epID); err != nil {
		t.Fatalf("Delete(%d) failed: %v", epID, err)
	}
	neighbors, err := index.Search(testVector(1), 1)
	if err != nil {
		t.Fatalf("Search failed after Delete following BulkDelete: %v", err)
	}
	if len(neighbors) != 1 {
		t.Fatalf("expected one neighbor, got %v", neighbors)
	}
	if _, ok := index.Nodes[neighbors[0].ID]; !ok {
		t.Errorf("search returned an id that is not in the index: %d", neighbors[0].ID)
	}
}

func TestHNSWIndex_BulkAddConcurrentWithAdd(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")

	vectors := make(map[int][]float32, 100)
	for i := 1000; i < 1100; i++ {
		vectors[i] = testVector(i)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		if err := index.BulkAdd(vectors); err != nil {
			t.Errorf("BulkAdd failed: %v", err)
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			if err := index.Add(i, testVector(i)); err != nil {
				t.Errorf("Add failed: %v", err)
			}
		}
	}()
	wg.Wait()

	stats := index.Stats()
	if stats.Count != 200 {
		t.Errorf("expected count 200 after concurrent BulkAdd and Add, got %d", stats.Count)
	}
}

func TestHNSWIndex_UpdateEntryPoint(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 32, core.Euclidean, "euclidean")
	// Exhaustive search keeps the exact-match assertions below free of the
	// approximation error of the greedy search.
	index.ExhaustiveSearch = true
	for i := 1; i <= 10; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Move the entry point node to a new location next to the others.
	epID := index.EntryPoint.ID
	if err := index.Update(epID, testVector(11)); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Every other node must still be reachable with an exact query.
	for i := 1; i <= 10; i++ {
		if i == epID {
			continue
		}
		neighbors, err := index.Search(testVector(i), 1)
		if err != nil {
			t.Fatalf("Search failed after updating the entry point: %v", err)
		}
		if len(neighbors) != 1 || neighbors[0].ID != i {
			t.Errorf("expected id %d as nearest neighbor after updating the entry point %d, got %v",
				i, epID, neighbors)
		}
	}

	// The updated node must be reachable at its new location.
	neighbors, err := index.Search(testVector(11), 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != epID {
		t.Errorf("expected id %d at its new location, got %v", epID, neighbors)
	}
}

func TestHNSWIndex_BulkUpdateGraphConsistency(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 32, core.Euclidean, "euclidean")
	// Exhaustive search keeps the exact-match assertions below free of the
	// approximation error of the greedy search.
	index.ExhaustiveSearch = true
	vectors := make(map[int][]float32, 20)
	for i := 1; i <= 20; i++ {
		vectors[i] = testVector(i)
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Move three nodes to new locations next to the others, so exact
	// queries remain answerable by an approximate index.
	updates := map[int][]float32{
		1: testVector(21),
		2: testVector(22),
		3: testVector(23),
	}
	if err := index.BulkUpdate(updates); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	// The graph must stay clean: no node may link to itself, and no node
	// may appear twice in another node's neighbor list at any level.
	for id, node := range index.Nodes {
		for level, links := range node.Links {
			seen := make(map[int]bool)
			for _, nb := range links {
				if nb.ID == id {
					t.Errorf("node %d links to itself at level %d after BulkUpdate", id, level)
				}
				if seen[nb.ID] {
					t.Errorf("node %d lists neighbor %d twice at level %d after BulkUpdate",
						id, nb.ID, level)
				}
				seen[nb.ID] = true
			}
		}
	}

	// Both updated and untouched nodes must be found at their locations.
	// Asking for every node engages the brute-force fallback, so the check
	// does not depend on the recall of the graph search.
	expected := map[int]int{1: 21, 2: 22, 3: 23}
	for i := 1; i <= 20; i++ {
		loc := i
		if moved, ok := expected[i]; ok {
			loc = moved
		}
		neighbors, err := index.Search(testVector(loc), 20)
		if err != nil {
			t.Fatalf("Search failed after BulkUpdate: %v", err)
		}
		if len(neighbors) == 0 || neighbors[0].ID != i {
			t.Errorf("expected id %d as nearest neighbor after BulkUpdate, got %v", i, neighbors)
		}
	}
}

func TestHNSWIndex_SearchSingleNodeLargeK(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10, core.Euclidean, "euclidean")
	if err := index.Add(1, testVector(1)); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Asking for more neighbors than the index holds must not panic.
	neighbors, err := index.Search(testVector(1), 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != 1 {
		t.Errorf("expected the single node as the only neighbor, got %v", neighbors)
	}
}

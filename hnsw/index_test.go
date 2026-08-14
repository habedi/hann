package hnsw_test

import (
	"bytes"
	"fmt"
	"os"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/hnsw"
	"github.com/habedi/hann/internal/testutil"
)

// newTestIndex constructs an index and fails the test when construction
// errors, so tests with known-good parameters stay short.
func newTestIndex(t *testing.T, dim int, opts ...hnsw.Option) *hnsw.Index {
	t.Helper()
	index, err := hnsw.New(dim, opts...)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	return index
}

func TestHNSWIndex_NewValidation(t *testing.T) {
	// A valid call succeeds with defaults.
	if _, err := hnsw.New(4); err != nil {
		t.Fatalf("New with defaults failed: %v", err)
	}

	cases := []struct {
		name string
		dim  int
		opts []hnsw.Option
	}{
		{"zero dimension", 0, nil},
		{"negative dimension", -1, nil},
		{"M below 2", 4, []hnsw.Option{hnsw.WithM(1)}},
		{"Ef below 1", 4, []hnsw.Option{hnsw.WithEf(0)}},
		{"zero metric", 4, []hnsw.Option{hnsw.WithMetric(core.Metric{})}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := hnsw.New(tc.dim, tc.opts...); err == nil {
				t.Errorf("expected error for %s, got none", tc.name)
			}
		})
	}
}

func TestHNSWIndex_AddAndStats(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

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
	if stats.Distance != core.Euclidean.Name() {
		t.Errorf("expected distance %q in stats, got %q", core.Euclidean.Name(), stats.Distance)
	}
}

func TestHNSWIndex_Delete(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

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
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

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
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

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
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

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
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

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
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

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
	newIndex := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
	if err := newIndex.Load(readFile); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Assert: check that stats match.
	stats := newIndex.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after Load, got %d", len(vectors), stats.Count)
	}
}

// testVector returns a 6-dimensional vector derived from i, spaced far
// apart so that nearest-neighbor results are unambiguous.
func testVector(i int) []float32 {
	base := float32(i * 100)
	return []float32{base, base + 1, base + 2, base + 3, base + 4, base + 5}
}

func TestHNSWIndex_SaveLoadEntryPointZero(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
	if err := index.Add(0, testVector(0)); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	var buf bytes.Buffer
	if err := index.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
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
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
	for i := 1; i <= 10; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	var buf bytes.Buffer
	if err := index.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load into a zero-value index: the metric and the internal maps must be
	// restored from the serialized state.
	loaded := &hnsw.Index{}
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
	custom := core.NewMetric("custom_metric", func(a, b []float32) (float64, error) {
		return core.Euclidean.Distance(a, b)
	}, false)
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10), hnsw.WithMetric(custom))
	for i := 1; i <= 5; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	var buf bytes.Buffer
	if err := index.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Loading into a zero-value index must fail, because the metric name is
	// unknown and no metric is present on the receiver.
	saved := buf.Bytes()
	empty := &hnsw.Index{}
	if err := empty.Load(bytes.NewReader(saved)); err == nil {
		t.Error("expected error when loading an unknown metric into a zero-value index, got none")
	}

	// Loading onto an index constructed with the custom metric must keep it.
	target := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10), hnsw.WithMetric(custom))
	if err := target.Load(bytes.NewReader(saved)); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	neighbors, err := target.Search(testVector(2), 1)
	if err != nil {
		t.Fatalf("Search failed after Load with a custom metric: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != 2 {
		t.Errorf("expected id 2 as nearest neighbor after Load, got %v", neighbors)
	}
}

func TestHNSWIndex_DeleteEntryPointAfterLoad(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
	for i := 1; i <= 10; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	var buf bytes.Buffer
	if err := index.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	loaded := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
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
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
	vectors := make(map[int][]float32, 50)
	for i := 1; i <= 50; i++ {
		vectors[i] = testVector(i)
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Delete every id but the last, one at a time, so whichever node is the
	// current entry point is deleted at some step. The survivors must remain
	// reachable after each deletion, which requires BulkAdd to have
	// maintained the level bookkeeping that Delete uses to pick a new entry
	// point.
	for i := 1; i <= 49; i++ {
		if err := index.Delete(i); err != nil {
			t.Fatalf("Delete(%d) failed: %v", i, err)
		}
		neighbors, err := index.Search(testVector(50), 1)
		if err != nil {
			t.Fatalf("Search failed with %d nodes left after Delete(%d): %v", 50-i, i, err)
		}
		if len(neighbors) != 1 || neighbors[0].ID <= i || neighbors[0].ID > 50 {
			t.Errorf("expected a surviving id after Delete(%d), got %v", i, neighbors)
		}
	}
}

func TestHNSWIndex_DeleteAfterBulkDelete(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
	for i := 1; i <= 100; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// Bulk delete half the nodes, then delete the remaining ones one at a
	// time. Whichever node is the entry point after BulkDelete is deleted at
	// some step, so Delete must pick a surviving node as the new entry
	// point, not a node that BulkDelete already removed from its level
	// bookkeeping.
	var deleteIDs []int
	for i := 1; i <= 50; i++ {
		deleteIDs = append(deleteIDs, i)
	}
	if err := index.BulkDelete(deleteIDs); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}

	for i := 51; i <= 99; i++ {
		if err := index.Delete(i); err != nil {
			t.Fatalf("Delete(%d) failed after BulkDelete: %v", i, err)
		}
		neighbors, err := index.Search(testVector(100), 1)
		if err != nil {
			t.Fatalf("Search failed after Delete(%d) following BulkDelete: %v", i, err)
		}
		if len(neighbors) != 1 {
			t.Fatalf("expected one neighbor after Delete(%d), got %v", i, neighbors)
		}
		if neighbors[0].ID <= i || neighbors[0].ID > 100 {
			t.Errorf("search returned an id that is not in the index after Delete(%d): %d",
				i, neighbors[0].ID)
		}
	}
}

func TestHNSWIndex_UpdateEntryPoint(t *testing.T) {
	dim := 6
	// Exhaustive search keeps the exact-match assertions below free of the
	// approximation error of the greedy search.
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(32), hnsw.WithExhaustiveSearch(true))
	for i := 1; i <= 10; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// The entry point is not observable through the public API, so update
	// every node in turn; whichever node is the entry point is covered.
	locations := make(map[int]int, 10)
	for i := 1; i <= 10; i++ {
		locations[i] = i
	}
	for i := 1; i <= 10; i++ {
		if err := index.Update(i, testVector(i+10)); err != nil {
			t.Fatalf("Update(%d) failed: %v", i, err)
		}
		locations[i] = i + 10

		// Every node must be reachable at its current location with an
		// exact query after each update.
		for id, loc := range locations {
			neighbors, err := index.Search(testVector(loc), 1)
			if err != nil {
				t.Fatalf("Search failed after Update(%d): %v", i, err)
			}
			if len(neighbors) != 1 || neighbors[0].ID != id {
				t.Errorf("expected id %d as nearest neighbor after Update(%d), got %v",
					id, i, neighbors)
			}
		}
	}
}

func TestHNSWIndex_BulkUpdateGraphConsistency(t *testing.T) {
	dim := 6
	// Exhaustive search keeps the exact-match assertions below free of the
	// approximation error of the greedy search.
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(32), hnsw.WithExhaustiveSearch(true))
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

	// Both updated and untouched nodes must be found at their locations.
	// Asking for every node engages the brute-force fallback, so the check
	// does not depend on the recall of the graph search. A corrupted graph
	// (self-links or duplicated neighbor entries) would surface here as a
	// duplicated or missing id in the full result set.
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
		seen := make(map[int]bool, len(neighbors))
		for _, n := range neighbors {
			if n.ID < 1 || n.ID > 20 {
				t.Errorf("search returned an id that is not in the index after BulkUpdate: %d", n.ID)
			}
			if seen[n.ID] {
				t.Errorf("search returned id %d twice after BulkUpdate", n.ID)
			}
			seen[n.ID] = true
		}
	}
}

func TestHNSWIndex_SearchSingleNodeLargeK(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
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

func TestHNSWIndex_SearchFallbackChunking(t *testing.T) {
	// The fallback path splits the unvisited nodes into chunks of
	// ceil(len/numWorkers). When the workers overshoot the slice by more than
	// one chunk, a worker's start index passes the slice length, which used to
	// panic with a slice bounds error. Sweeping the index size across several
	// multiples of the worker count hits the bad remainder on any CPU count
	// that has one.
	dim := 4
	for n := 3; n <= 128; n++ {
		// A small ef keeps searchLayer's result list short, so a search with
		// k equal to the node count always enters the fallback.
		index := newTestIndex(t, dim, hnsw.WithM(4), hnsw.WithEf(2))
		for i := 0; i < n; i++ {
			vec := []float32{float32(i), float32(i % 7), float32(i % 11), float32(i % 13)}
			if err := index.Add(i, vec); err != nil {
				t.Fatalf("Add(%d) failed: %v", i, err)
			}
		}
		neighbors, err := index.Search([]float32{0, 0, 0, 0}, n)
		if err != nil {
			t.Fatalf("Search failed with %d nodes: %v", n, err)
		}
		if len(neighbors) != n {
			t.Fatalf("Search with k=%d returned %d neighbors", n, len(neighbors))
		}
	}
}

// hnswFactory describes the HNSW index for the shared test runners. Search
// sorts its candidates before returning, so results are sorted, and the
// reported distances are computed against the stored vectors, so they are
// exact.
func hnswFactory() testutil.Factory {
	return testutil.Factory{
		New: func() core.Index {
			index, err := hnsw.New(16, hnsw.WithM(16), hnsw.WithEf(100))
			if err != nil {
				panic(fmt.Sprintf("hnsw.New failed: %v", err))
			}
			return index
		},
		ExactDistances: true,
		SortedResults:  true,
		Metric:         core.Euclidean,
	}
}

// TestHNSWIndex_FallbackSearchCounter checks that a search that falls back to
// a brute-force scan is visible in Stats.
func TestHNSWIndex_FallbackSearchCounter(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10))
	if err := index.Add(1, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if got := index.Stats().FallbackSearches; got != 0 {
		t.Fatalf("expected 0 fallback searches before any search, got %d", got)
	}
	// k larger than the number of reachable candidates forces the fallback.
	if _, err := index.Search([]float32{1, 2, 3, 4}, 2); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if got := index.Stats().FallbackSearches; got < 1 {
		t.Fatalf("expected at least 1 fallback search, got %d", got)
	}
}

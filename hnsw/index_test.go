package hnsw_test

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/hnsw"
	"github.com/habedi/hann/internal/testutil"
)

// newTestIndex builds an index and fails the test when construction
// errors. This keeps tests with known-good parameters short.
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
		{"EfConstruction below 1", 4, []hnsw.Option{hnsw.WithEfConstruction(0)}},
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

	// Add two vectors.
	if err := index.Add(1, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if err := index.Add(2, []float32{6, 5, 4, 3, 2, 1}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Delete id 1.
	if err := index.Delete(1); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	// The count must now be 1.
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

	// Add a vector.
	if err := index.Add(1, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Update it with a new vector.
	if err := index.Update(1, []float32{6, 6, 6, 6, 6, 6}); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Search with the updated vector.
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

	// Create a set of 5 vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {2, 2, 2, 2, 2, 2},
		5: {3, 3, 3, 3, 3, 3},
	}

	// Bulk add the vectors.
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Check the index count.
	stats := index.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after BulkAdd, got %d", len(vectors), stats.Count)
	}
}

func TestHNSWIndex_BulkDelete(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

	// Bulk add a set of vectors.
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

	// Bulk delete some ids.
	deleteIDs := []int{2, 4}
	if err := index.BulkDelete(deleteIDs); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}

	// Verify the count and ensure the deleted ids are gone.
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

	// Bulk add a set of vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Bulk update vectors for some ids.
	updates := map[int][]float32{
		1: {6, 6, 6, 6, 6, 6},
		3: {2, 2, 2, 2, 2, 2},
	}
	if err := index.BulkUpdate(updates); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	// Search with a query similar to the updated vector of id 1.
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

	// Add some vectors.
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

	// Check that the stats match.
	stats := newIndex.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after Load, got %d", len(vectors), stats.Count)
	}
}

// testVector returns a 6-dimensional vector derived from i. The vectors
// are spaced far apart, so nearest-neighbor results are unambiguous.
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

	// Delete all ids but one. After each deletion the survivors must still
	// be reachable. That only works when Load has rebuilt the level
	// bookkeeping.
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

	// Delete every id but the last, one at a time. This way whichever node
	// is the current entry point is deleted at some step. The survivors
	// must remain reachable after each deletion. That only works when
	// BulkAdd has maintained the level bookkeeping that Delete uses to pick
	// a new entry point.
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
	// time. Whichever node is the entry point after BulkDelete is deleted
	// at some step. Delete must then pick a surviving node as the new entry
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
	// Exhaustive search removes the approximation error of the greedy
	// search from the exact-match assertions below.
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(32), hnsw.WithExhaustiveSearch(true))
	for i := 1; i <= 10; i++ {
		if err := index.Add(i, testVector(i)); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	// The entry point is not observable through the public API, so update
	// every node in turn. Whichever node is the entry point is covered.
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
	// Exhaustive search removes the approximation error of the greedy
	// search from the exact-match assertions below.
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(32), hnsw.WithExhaustiveSearch(true))
	vectors := make(map[int][]float32, 20)
	for i := 1; i <= 20; i++ {
		vectors[i] = testVector(i)
	}
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Move three nodes to new locations next to the others. This keeps
	// exact queries answerable by an approximate index.
	updates := map[int][]float32{
		1: testVector(21),
		2: testVector(22),
		3: testVector(23),
	}
	if err := index.BulkUpdate(updates); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	// Both updated and untouched nodes must be found at their locations.
	// Asking for every node triggers the brute-force fallback, so the check
	// does not depend on the recall of the graph search. A corrupted graph
	// (self-links or duplicated neighbor entries) would show up here as a
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
	// ceil(len/numWorkers). When the workers overshoot the slice by more
	// than one chunk, a worker's start index passes the slice length. That
	// used to panic with a slice bounds error. Sweeping the index size
	// across several multiples of the worker count hits the bad remainder
	// on any CPU count that has one.
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
// sorts its candidates before returning, so results are sorted. The
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

func TestHNSWIndex_SearchArgumentErrors(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10))

	// A search on an empty index must fail.
	if _, err := index.Search([]float32{1, 2, 3, 4}, 1); err == nil {
		t.Error("expected error searching an empty index, got none")
	}

	if err := index.Add(1, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// A query with the wrong dimension must fail.
	if _, err := index.Search([]float32{1, 2}, 1); err == nil {
		t.Error("expected error for a query with the wrong dimension, got none")
	}
}

// TestHNSWIndex_SearchFallbackClampsK checks a k larger than the index.
// The search must still return every node exactly once, in sorted order,
// after the brute-force fallback has merged its candidates.
func TestHNSWIndex_SearchFallbackClampsK(t *testing.T) {
	// A small ef keeps searchLayer's result list short. The search then
	// enters the fallback and gathers the remaining nodes there.
	index := newTestIndex(t, 4, hnsw.WithM(4), hnsw.WithEf(2))
	for i := 1; i <= 5; i++ {
		if err := index.Add(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatalf("Add(%d) failed: %v", i, err)
		}
	}
	neighbors, err := index.Search([]float32{0, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 5 {
		t.Fatalf("expected 5 neighbors for k=10 over 5 nodes, got %d", len(neighbors))
	}
	for i, n := range neighbors {
		if n.ID != i+1 {
			t.Errorf("rank %d: expected id %d, got %d", i, i+1, n.ID)
		}
		if i > 0 && n.Distance < neighbors[i-1].Distance {
			t.Errorf("rank %d: results are not sorted", i)
		}
	}
}

// TestHNSWIndex_SearchFallbackSmallShortfall checks the fallback with a
// shortfall smaller than the number of unvisited nodes. The per-worker
// heaps and the merge heap must then push out worse candidates instead of
// only collecting them. The final result must stay deduplicated and
// sorted.
func TestHNSWIndex_SearchFallbackSmallShortfall(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithM(4), hnsw.WithEf(2))
	const n = 60
	for i := 1; i <= n; i++ {
		if err := index.Add(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatalf("Add(%d) failed: %v", i, err)
		}
	}
	// With ef 2 the layer search returns at most 2 candidates, so k 3
	// leaves a shortfall of 1. The parallel scan must fill it with the
	// single best remaining node.
	neighbors, err := index.Search([]float32{float32(n + 1), 0, 0, 0}, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 3 {
		t.Fatalf("expected 3 neighbors, got %d", len(neighbors))
	}
	seen := make(map[int]bool, len(neighbors))
	for i, nb := range neighbors {
		if nb.ID < 1 || nb.ID > n {
			t.Errorf("rank %d: id %d is not in the index", i, nb.ID)
		}
		if seen[nb.ID] {
			t.Errorf("rank %d: id %d returned twice", i, nb.ID)
		}
		seen[nb.ID] = true
		if i > 0 && nb.Distance < neighbors[i-1].Distance {
			t.Errorf("rank %d: results are not sorted", i)
		}
	}
	if neighbors[0].ID != n {
		t.Errorf("expected the nearest id %d first, got %v", n, neighbors)
	}
}

func TestHNSWIndex_BulkAddValidation(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10))
	if err := index.Add(1, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// A vector with the wrong dimension must fail.
	if err := index.BulkAdd(map[int][]float32{2: {1, 2}}); err == nil {
		t.Error("expected error for a bulk vector with the wrong dimension, got none")
	}

	// An id that already exists must fail.
	if err := index.BulkAdd(map[int][]float32{1: {5, 6, 7, 8}}); err == nil {
		t.Error("expected error for a duplicate id in BulkAdd, got none")
	}

	// The failed calls must not have added anything.
	if got := index.Stats().Count; got != 1 {
		t.Errorf("expected count 1 after failed BulkAdd calls, got %d", got)
	}
}

// TestHNSWIndex_BulkDeleteUnknownID checks that an unknown id is skipped and
// the known ids are still removed.
func TestHNSWIndex_BulkDeleteUnknownID(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10))
	if err := index.BulkAdd(map[int][]float32{
		1: {1, 0, 0, 0},
		2: {2, 0, 0, 0},
	}); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := index.BulkDelete([]int{99, 2}); err != nil {
		t.Fatalf("BulkDelete with an unknown id failed: %v", err)
	}
	if got := index.Stats().Count; got != 1 {
		t.Fatalf("expected count 1 after BulkDelete, got %d", got)
	}
	neighbors, err := index.Search([]float32{2, 0, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != 1 {
		t.Errorf("expected the surviving id 1, got %v", neighbors)
	}
}

// TestHNSWIndex_BulkUpdateWrongDimension checks that a wrong-dimension
// vector for an existing id fails the batch. The entry must stay at its old
// location.
func TestHNSWIndex_BulkUpdateWrongDimension(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10))
	if err := index.BulkAdd(map[int][]float32{
		1: {1, 0, 0, 0},
		2: {2, 0, 0, 0},
	}); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := index.BulkUpdate(map[int][]float32{1: {9, 9}}); err == nil {
		t.Fatal("expected error for a bulk update with the wrong dimension, got none")
	}
	neighbors, err := index.Search([]float32{1, 0, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].ID != 1 {
		t.Errorf("expected id 1 at its old location after the failed BulkUpdate, got %v", neighbors)
	}
}

// TestHNSWIndex_CosineOperations exercises the normalization paths of Add,
// Update, BulkAdd, and BulkUpdate under the cosine metric, including the
// dimension checks inside the batch normalization loops.
func TestHNSWIndex_CosineOperations(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10), hnsw.WithMetric(core.Cosine))

	// BulkAdd normalizes its vectors before insertion.
	if err := index.BulkAdd(map[int][]float32{
		1: {2, 0, 0, 0},
		2: {0, 3, 0, 0},
	}); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	// The dimension check inside the normalization loop must reject a bad row.
	if err := index.BulkAdd(map[int][]float32{3: {1, 2}}); err == nil {
		t.Error("expected error for a wrong-dimension vector in cosine BulkAdd, got none")
	}

	// Add normalizes a single vector.
	if err := index.Add(3, []float32{0, 0, 4, 0}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Update normalizes the replacement vector.
	if err := index.Update(3, []float32{0, 0, 0, 7}); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// BulkUpdate normalizes its vectors. Its normalization loop rejects a
	// wrong-dimension row before any entry is touched.
	if err := index.BulkUpdate(map[int][]float32{1: {1, 2}}); err == nil {
		t.Error("expected error for a wrong-dimension vector in cosine BulkUpdate, got none")
	}
	if err := index.BulkUpdate(map[int][]float32{1: {8, 0, 0, 0}}); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	// Under cosine only the direction matters. Each direction must map to
	// its node regardless of the stored magnitudes.
	cases := map[int][]float32{
		1: {1, 0, 0, 0},
		2: {0, 1, 0, 0},
		3: {0, 0, 0, 1},
	}
	for id, query := range cases {
		neighbors, err := index.Search(query, 1)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(neighbors) != 1 || neighbors[0].ID != id {
			t.Errorf("expected id %d for its direction, got %v", id, neighbors)
		}
	}
}

// newFlakyIndex returns an index whose metric behaves like the Euclidean
// distance until the returned flag is set. After that every distance call
// fails. The flag is only flipped between operations, so it needs no
// locking.
func newFlakyIndex(t *testing.T) (*hnsw.Index, *bool) {
	t.Helper()
	failNow := false
	metric := core.NewMetric("flaky_test_metric", func(a, b []float32) (float64, error) {
		if failNow {
			return 0, fmt.Errorf("distance failure injected by the test")
		}
		return core.Euclidean.Distance(a, b)
	}, false)
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10), hnsw.WithMetric(metric))
	for i := 1; i <= 2; i++ {
		if err := index.Add(i, []float32{float32(i), 0, 0, 0}); err != nil {
			t.Fatalf("Add(%d) failed: %v", i, err)
		}
	}
	return index, &failNow
}

// TestHNSWIndex_MetricErrorPropagation checks that a distance failure
// during graph maintenance is returned to the caller. A failed Add must
// roll the new node back out of the index.
func TestHNSWIndex_MetricErrorPropagation(t *testing.T) {
	t.Run("Add rolls back", func(t *testing.T) {
		index, failNow := newFlakyIndex(t)
		*failNow = true
		if err := index.Add(3, []float32{3, 0, 0, 0}); err == nil {
			t.Fatal("expected error from Add with a failing metric, got none")
		}
		if got := index.Stats().Count; got != 2 {
			t.Fatalf("expected count 2 after the rolled-back Add, got %d", got)
		}
		// The rollback must leave the id free for a later insert.
		*failNow = false
		if err := index.Add(3, []float32{3, 0, 0, 0}); err != nil {
			t.Fatalf("Add after the rollback failed: %v", err)
		}
		neighbors, err := index.Search([]float32{3, 0, 0, 0}, 1)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(neighbors) != 1 || neighbors[0].ID != 3 {
			t.Errorf("expected id 3 after the re-added insert, got %v", neighbors)
		}
	})

	t.Run("Update returns the error", func(t *testing.T) {
		index, failNow := newFlakyIndex(t)
		*failNow = true
		if err := index.Update(1, []float32{9, 0, 0, 0}); err == nil {
			t.Error("expected error from Update with a failing metric, got none")
		}
	})

	t.Run("BulkUpdate returns the error", func(t *testing.T) {
		index, failNow := newFlakyIndex(t)
		*failNow = true
		if err := index.BulkUpdate(map[int][]float32{1: {9, 0, 0, 0}}); err == nil {
			t.Error("expected error from BulkUpdate with a failing metric, got none")
		}
	})

	t.Run("BulkAdd returns the error", func(t *testing.T) {
		index, failNow := newFlakyIndex(t)
		*failNow = true
		if err := index.BulkAdd(map[int][]float32{3: {3, 0, 0, 0}}); err == nil {
			t.Error("expected error from BulkAdd with a failing metric, got none")
		}
	})

	t.Run("Search returns the error", func(t *testing.T) {
		index, failNow := newFlakyIndex(t)
		*failNow = true
		if _, err := index.Search([]float32{1, 0, 0, 0}, 1); err == nil {
			t.Error("expected error from Search with a failing metric, got none")
		}
	})
}

// failingWriter fails every write, so Save must surface the writer's error.
type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) {
	return 0, fmt.Errorf("write failure injected by the test")
}

func TestHNSWIndex_SaveFailingWriter(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10))
	if err := index.Add(1, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if err := index.Save(failingWriter{}); err == nil {
		t.Error("expected error from Save with a failing writer, got none")
	}
}

func TestHNSWIndex_GobDecodeErrors(t *testing.T) {
	// Bytes that are not a gob stream must fail.
	index := newTestIndex(t, 4, hnsw.WithM(5), hnsw.WithEf(10))
	if err := index.GobDecode([]byte("not a gob stream")); err == nil {
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
	err := index.GobDecode(buf.Bytes())
	if err == nil {
		t.Fatal("expected error for a newer format version, got none")
	}
	if !strings.Contains(err.Error(), "format version") {
		t.Errorf("expected a format version error, got %v", err)
	}
}

// legacyHNSWNode and legacyHNSWIndex mirror the on-disk shape written
// before the HasEntryPoint, FormatVersion, and EfConstruction fields
// existed. The gob decoder matches struct fields by name. Encoding them
// therefore produces the byte stream an old version of the library would
// have written.
type legacyHNSWNode struct {
	ID     int
	Vector []float32
	Level  int
	Links  map[int][]int
}

type legacyHNSWIndex struct {
	Dimension        int
	M                int
	Ef               int
	Nodes            map[int]legacyHNSWNode
	EntryPoint       int
	MaxLevel         int
	DistanceName     string
	ExhaustiveSearch bool
}

// decodeLegacyHNSW encodes the legacy form and decodes it into a fresh index.
func decodeLegacyHNSW(t *testing.T, legacy legacyHNSWIndex) *hnsw.Index {
	t.Helper()
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(legacy); err != nil {
		t.Fatalf("encoding the legacy index failed: %v", err)
	}
	index := &hnsw.Index{}
	if err := index.GobDecode(buf.Bytes()); err != nil {
		t.Fatalf("GobDecode of the legacy index failed: %v", err)
	}
	return index
}

// TestHNSWIndex_GobDecodeLegacyFormat checks that files written before the
// HasEntryPoint flag existed still load: a nonzero entry point id is used
// directly, the id 0 sentinel recovers the node with id 0, and an empty
// file leaves the index empty.
func TestHNSWIndex_GobDecodeLegacyFormat(t *testing.T) {
	t.Run("nonzero entry point", func(t *testing.T) {
		legacy := legacyHNSWIndex{
			Dimension: 6, M: 5, Ef: 10,
			Nodes: map[int]legacyHNSWNode{
				1: {ID: 1, Vector: testVector(1), Level: 0, Links: map[int][]int{0: {2}}},
				2: {ID: 2, Vector: testVector(2), Level: 0, Links: map[int][]int{0: {1, 3}}},
				3: {ID: 3, Vector: testVector(3), Level: 0, Links: map[int][]int{0: {2}}},
			},
			EntryPoint: 2, MaxLevel: 0, DistanceName: "euclidean",
		}
		index := decodeLegacyHNSW(t, legacy)
		neighbors, err := index.Search(testVector(1), 1)
		if err != nil {
			t.Fatalf("Search failed after decoding a legacy index: %v", err)
		}
		if len(neighbors) != 1 || neighbors[0].ID != 1 {
			t.Errorf("expected id 1 as nearest neighbor, got %v", neighbors)
		}
	})

	t.Run("entry point with id 0", func(t *testing.T) {
		legacy := legacyHNSWIndex{
			Dimension: 6, M: 5, Ef: 10,
			Nodes: map[int]legacyHNSWNode{
				0: {ID: 0, Vector: testVector(0), Level: 0, Links: map[int][]int{0: {1}}},
				1: {ID: 1, Vector: testVector(1), Level: 0, Links: map[int][]int{0: {0}}},
			},
			EntryPoint: 0, MaxLevel: 0, DistanceName: "euclidean",
		}
		index := decodeLegacyHNSW(t, legacy)
		neighbors, err := index.Search(testVector(0), 1)
		if err != nil {
			t.Fatalf("Search failed after decoding a legacy index: %v", err)
		}
		if len(neighbors) != 1 || neighbors[0].ID != 0 {
			t.Errorf("expected id 0 as nearest neighbor, got %v", neighbors)
		}
	})

	t.Run("empty index", func(t *testing.T) {
		legacy := legacyHNSWIndex{
			Dimension: 6, M: 5, Ef: 10,
			MaxLevel: -1, DistanceName: "euclidean",
		}
		index := decodeLegacyHNSW(t, legacy)
		if got := index.Stats().Count; got != 0 {
			t.Fatalf("expected an empty index, got count %d", got)
		}
		if _, err := index.Search(testVector(1), 1); err == nil {
			t.Error("expected error searching a decoded empty index, got none")
		}
	})
}

func TestHNSWIndex_SetEf(t *testing.T) {
	index := newTestIndex(t, 4, hnsw.WithEf(10))
	if err := index.SetEf(200); err != nil {
		t.Fatalf("SetEf failed: %v", err)
	}
	if err := index.SetEf(0); err == nil {
		t.Fatal("expected error for non-positive ef, got none")
	}
	// The index stays usable after the change.
	if err := index.Add(1, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if _, err := index.Search([]float32{1, 2, 3, 4}, 1); err != nil {
		t.Fatalf("Search failed: %v", err)
	}
}

// newCountdownMetric returns a metric that works like the Euclidean
// distance until the countdown reaches zero, after which every call fails.
// A countdown below zero never fails. The counter is atomic, because the
// fallback scan calls the metric from several goroutines.
func newCountdownMetric() (core.Metric, *atomic.Int64) {
	countdown := new(atomic.Int64)
	countdown.Store(-1)
	metric := core.NewMetric("countdown_test_metric", func(a, b []float32) (float64, error) {
		c := countdown.Load()
		if c == 0 {
			return 0, fmt.Errorf("distance failure injected by the test")
		}
		if c > 0 {
			countdown.Add(-1)
		}
		return core.Euclidean.Distance(a, b)
	}, false)
	return metric, countdown
}

// newCountdownIndex returns an index of n random vectors built with the
// countdown metric, together with the metric and its counter. Loading the
// returned snapshot bytes into an index configured with the same metric
// restores the same graph, because the metric's name is not registered and
// the decode keeps the configured metric.
func newCountdownIndex(t *testing.T, dim, n int) (*hnsw.Index, core.Metric, *atomic.Int64) {
	t.Helper()
	metric, countdown := newCountdownMetric()
	idx := newTestIndex(t, dim,
		hnsw.WithM(4), hnsw.WithEf(30), hnsw.WithEfConstruction(20), hnsw.WithMetric(metric))
	rng := rand.New(rand.NewSource(11))
	for id := 0; id < n; id++ {
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add(%d) failed: %v", id, err)
		}
	}
	return idx, metric, countdown
}

// saveBytes serializes the index, so a test can restore the same graph many
// times without rebuilding it.
func saveBytes(t *testing.T, idx *hnsw.Index) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	return buf.Bytes()
}

// allIDs runs a complete search and returns the ids it finds.
func allIDs(t *testing.T, idx *hnsw.Index, query []float32) map[int]bool {
	t.Helper()
	count := idx.Stats().Count
	neighbors, err := idx.Search(query, count)
	if err != nil {
		t.Fatalf("complete Search failed: %v", err)
	}
	ids := make(map[int]bool, len(neighbors))
	for _, nb := range neighbors {
		ids[nb.ID] = true
	}
	return ids
}

// TestHNSWIndex_FailedAddLeavesNoTrace injects a metric failure at every
// possible point of an insertion. After each failed Add, the id must be
// fully gone: the count unchanged, no search may return it, and the id must
// be free for a later insert. The sweep stops at the first countdown that no
// longer fails, which means every failure point was covered.
func TestHNSWIndex_FailedAddLeavesNoTrace(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim, n := 8, 30
	base, metric, countdown := newCountdownIndex(t, dim, n)
	snapshot := saveBytes(t, base)
	ghost := make([]float32, dim)
	for i := range ghost {
		ghost[i] = 50
	}
	completed := false
	for cd := int64(1); cd <= 5000; cd++ {
		idx := newTestIndex(t, dim, hnsw.WithMetric(metric))
		if err := idx.Load(bytes.NewReader(snapshot)); err != nil {
			t.Fatalf("Load failed: %v", err)
		}
		countdown.Store(cd)
		errAdd := idx.Add(999, ghost)
		countdown.Store(-1)
		if errAdd == nil {
			completed = true
			break
		}
		if got := idx.Stats().Count; got != n {
			t.Fatalf("countdown %d: count %d after the failed Add, want %d", cd, got, n)
		}
		if allIDs(t, idx, ghost)[999] {
			t.Fatalf("countdown %d: a search returned the rolled-back id 999", cd)
		}
		if err := idx.Add(999, ghost); err != nil {
			t.Fatalf("countdown %d: re-adding id 999 after the rollback failed: %v", cd, err)
		}
		// A complete search runs the fallback scan, so it sees the re-added
		// id regardless of the shape the trims left the graph in.
		if !allIDs(t, idx, ghost)[999] {
			t.Fatalf("countdown %d: id 999 missing after the re-add", cd)
		}
	}
	if !completed {
		t.Fatal("the sweep never reached a successful Add; raise the countdown limit")
	}
}

// TestHNSWIndex_FailedUpdateLeavesIndexUnchanged injects a metric failure at
// every possible point of an update. The core.Index contract says a failed
// update leaves the index unchanged, so after each failure the stored vector
// must be the old one, and a complete search must return the same ids and
// distances as before the attempt.
func TestHNSWIndex_FailedUpdateLeavesIndexUnchanged(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim, n := 8, 30
	base, metric, countdown := newCountdownIndex(t, dim, n)
	snapshot := saveBytes(t, base)
	moved := make([]float32, dim)
	for i := range moved {
		moved[i] = 50
	}
	fixedQuery := make([]float32, dim)
	fixedQuery[0] = 1
	completed := false
	for cd := int64(1); cd <= 5000; cd++ {
		idx := newTestIndex(t, dim, hnsw.WithMetric(metric))
		if err := idx.Load(bytes.NewReader(snapshot)); err != nil {
			t.Fatalf("Load failed: %v", err)
		}
		before, err := idx.Search(fixedQuery, n)
		if err != nil {
			t.Fatalf("countdown %d: baseline Search failed: %v", cd, err)
		}
		countdown.Store(cd)
		errUpdate := idx.Update(5, moved)
		countdown.Store(-1)
		if errUpdate == nil {
			completed = true
			break
		}
		after, err := idx.Search(fixedQuery, n)
		if err != nil {
			t.Fatalf("countdown %d: post-failure Search failed: %v", cd, err)
		}
		if len(after) != len(before) {
			t.Fatalf("countdown %d: result size changed from %d to %d", cd, len(before), len(after))
		}
		for i := range before {
			if before[i].ID != after[i].ID || before[i].Distance != after[i].Distance {
				t.Fatalf("countdown %d: rank %d changed from %v to %v after the failed update",
					cd, i, before[i], after[i])
			}
		}
	}
	if !completed {
		t.Fatal("the sweep never reached a successful Update; raise the countdown limit")
	}
}

// TestHNSWIndex_FailedBulkAddRollsBackFailedNode injects a metric failure at
// every possible point of a three-vector BulkAdd. Every batch id must
// afterward be either fully present (Delete succeeds) or fully absent (no
// search returns it). A ghost fails both.
func TestHNSWIndex_FailedBulkAddRollsBackFailedNode(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim, n := 8, 30
	base, metric, countdown := newCountdownIndex(t, dim, n)
	snapshot := saveBytes(t, base)
	batch := make(map[int][]float32, 3)
	for j := 0; j < 3; j++ {
		vec := make([]float32, dim)
		vec[j] = 50
		batch[900+j] = vec
	}
	completed := false
	for cd := int64(1); cd <= 8000; cd++ {
		idx := newTestIndex(t, dim, hnsw.WithMetric(metric))
		if err := idx.Load(bytes.NewReader(snapshot)); err != nil {
			t.Fatalf("Load failed: %v", err)
		}
		countdown.Store(cd)
		errBulk := idx.BulkAdd(batch)
		countdown.Store(-1)
		if errBulk == nil {
			completed = true
			break
		}
		present := 0
		for id, vec := range batch {
			delErr := idx.Delete(id)
			if delErr == nil {
				present++
			}
			if allIDs(t, idx, vec)[id] {
				t.Fatalf("countdown %d: id %d still searchable (Delete error: %v)", cd, id, delErr)
			}
		}
		// The elements inserted before the failure stay, but the failing
		// element itself must have been rolled back, so at least one batch
		// id has to be absent.
		if present == len(batch) {
			t.Fatalf("countdown %d: BulkAdd failed but every batch element is present", cd)
		}
		if got := idx.Stats().Count; got != n {
			t.Fatalf("countdown %d: count %d after deleting the batch, want %d", cd, got, n)
		}
	}
	if !completed {
		t.Fatal("the sweep never reached a successful BulkAdd; raise the countdown limit")
	}
}

// TestHNSWIndex_GobDecodeSanitizesParameters loads a file whose M and Ef are
// out of range, as a corrupt or crafted file's would be. The decode must
// fall back to the defaults: with the defaults in place, a small index
// answers a small search from the graph alone, without the brute-force
// fallback.
func TestHNSWIndex_GobDecodeSanitizesParameters(t *testing.T) {
	payload := struct {
		Dimension    int
		M            int
		Ef           int
		MaxLevel     int
		DistanceName string
	}{Dimension: 8, M: 0, Ef: 0, MaxLevel: -1, DistanceName: "euclidean"}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(payload); err != nil {
		t.Fatalf("encoding the payload failed: %v", err)
	}
	var idx hnsw.Index
	if err := idx.GobDecode(buf.Bytes()); err != nil {
		t.Fatalf("GobDecode failed: %v", err)
	}
	rng := rand.New(rand.NewSource(12))
	for id := 0; id < 80; id++ {
		vec := make([]float32, 8)
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add(%d) failed: %v", id, err)
		}
	}
	query := make([]float32, 8)
	got, err := idx.Search(query, 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(got) != 5 {
		t.Fatalf("expected 5 results, got %d", len(got))
	}
	if fb := idx.Stats().FallbackSearches; fb != 0 {
		t.Errorf("expected the graph to answer without the fallback, got %d fallback searches", fb)
	}
}

// TestHNSWLevelFromUnit checks the mapping from a uniform draw to a node
// level, including the r equal to 0 edge, whose logarithm is infinite.
func TestHNSWLevelFromUnit(t *testing.T) {
	if got := hnsw.LevelFromUnit(0, 16); got != hnsw.MaxLevelCap {
		t.Errorf("LevelFromUnit(0, 16) = %d, want the cap %d", got, hnsw.MaxLevelCap)
	}
	if got := hnsw.LevelFromUnit(1e-300, 16); got != hnsw.MaxLevelCap {
		t.Errorf("LevelFromUnit(1e-300, 16) = %d, want the cap %d", got, hnsw.MaxLevelCap)
	}
	if got := hnsw.LevelFromUnit(0.99, 16); got != 0 {
		t.Errorf("LevelFromUnit(0.99, 16) = %d, want 0", got)
	}
	for _, r := range []float64{0, 1e-300, 1e-9, 0.01, 0.25, 0.5, 0.999999} {
		got := hnsw.LevelFromUnit(r, 16)
		if got < 0 || got > hnsw.MaxLevelCap {
			t.Errorf("LevelFromUnit(%g, 16) = %d, out of [0, %d]", r, got, hnsw.MaxLevelCap)
		}
	}
}

// TestHNSWIndex_BulkAddGraphInvariants bulk-adds thousands of vectors, then
// audits the structural invariants of the graph: link counts within M, no
// duplicate or dangling links, a reverse record behind every link, and a
// valid entry point. The audit repeats after deletes and updates on the
// bulk-built graph, over several rounds so the insertion interleavings
// vary.
func TestHNSWIndex_BulkAddGraphInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for round := 0; round < 4; round++ {
		dim := 16
		idx := newTestIndex(t, dim, hnsw.WithM(8), hnsw.WithEf(40), hnsw.WithEfConstruction(60))
		vectors := make(map[int][]float32, 3000)
		for id := 0; id < 3000; id++ {
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = float32(rng.NormFloat64())
			}
			vectors[id] = vec
		}
		if err := idx.BulkAdd(vectors); err != nil {
			t.Fatalf("round %d: BulkAdd failed: %v", round, err)
		}
		for _, p := range hnsw.AuditGraph(idx) {
			t.Errorf("round %d after BulkAdd: %s", round, p)
		}
		if t.Failed() {
			return
		}

		var toDelete []int
		for id := 0; id < 3000; id += 3 {
			toDelete = append(toDelete, id)
		}
		if err := idx.BulkDelete(toDelete); err != nil {
			t.Fatalf("round %d: BulkDelete failed: %v", round, err)
		}
		updates := make(map[int][]float32, 500)
		for id := 1; id < 1500; id += 3 {
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = float32(rng.NormFloat64())
			}
			updates[id] = vec
		}
		if err := idx.BulkUpdate(updates); err != nil {
			t.Fatalf("round %d: BulkUpdate failed: %v", round, err)
		}
		for _, p := range hnsw.AuditGraph(idx) {
			t.Errorf("round %d after delete and update: %s", round, p)
		}
	}
}

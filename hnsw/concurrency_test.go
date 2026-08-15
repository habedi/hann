package hnsw_test

import (
	"io"
	"sync"
	"testing"
	"time"

	"github.com/habedi/hann/hnsw"
	"github.com/habedi/hann/internal/testutil"
)

func TestHNSWIndex_ConcurrentBulkOperations(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
	numVectors := 1000

	// Prepare a map of vectors.
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

	// Perform the BulkAdd.
	if err := index.BulkAdd(vectors); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Update half of the vectors.
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

	// Delete one-quarter of the vectors.
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

	// Check the final count.
	expected := numVectors - len(deleteIDs)
	stats := index.Stats()
	if stats.Count != expected {
		t.Errorf("expected count %d after concurrent bulk operations, got %d", expected,
			stats.Count)
	}
}

func TestHNSWIndex_SaveConcurrentWithAdd(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))
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

func TestHNSWIndex_BulkAddConcurrentWithAdd(t *testing.T) {
	dim := 6
	index := newTestIndex(t, dim, hnsw.WithM(5), hnsw.WithEf(10))

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

func TestHNSWIndex_ConcurrentOps(t *testing.T) {
	testutil.RunConcurrentOps(t, hnswFactory(), 16, 8, 300)
}

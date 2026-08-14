package rpt_test

import (
	"bytes"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/habedi/hann/internal/testutil"
	"github.com/habedi/hann/rpt"
)

func TestRPTIndex_ConcurrentOperations(t *testing.T) {
	dim := 6
	idx := mustNew(t, dim)
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

// TestRPTIndex_ConcurrentSearchWithWrites checks that Search does not read the
// point map without holding the lock while other goroutines mutate the index.
// Run with the race detector to catch unlocked map access.
func TestRPTIndex_ConcurrentSearchWithWrites(t *testing.T) {
	dim := 8
	idx := mustNew(t, dim,
		rpt.WithLeafCapacity(5),
		rpt.WithCandidateProjections(2),
		rpt.WithParallelThreshold(1<<30),
		rpt.WithProbeMargin(0.1))
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
	idx := mustNew(t, dim,
		rpt.WithLeafCapacity(5),
		rpt.WithCandidateProjections(2),
		rpt.WithParallelThreshold(1<<30),
		rpt.WithProbeMargin(0.1))
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
	idx := mustNew(t, dim,
		rpt.WithLeafCapacity(5),
		rpt.WithCandidateProjections(1),
		rpt.WithParallelThreshold(1<<30),
		rpt.WithProbeMargin(1e9))
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

// TestRPTIndex_ConcurrentStress exercises the index from many goroutines at
// once. Run with the race detector; this test is the permanent guard for the
// concurrency bugs the package had.
func TestRPTIndex_ConcurrentStress(t *testing.T) {
	testutil.RunConcurrentOps(t, rptFactory(t, 16), 16, 8, 300)
}

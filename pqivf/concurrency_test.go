package pqivf_test

import (
	"bytes"
	"sync"
	"testing"
	"time"

	"github.com/habedi/hann/internal/testutil"
)

func TestPQIVF_ConcurrentOperations(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	pqK := 256
	kMeansIters := 10
	idx := newIndex(t, dim, coarseK, numSubquantizers, pqK, kMeansIters)
	numVectors := 1000
	var wg sync.WaitGroup

	for i := range numVectors {
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

func TestPQIVF_ConcurrentSaveAdd(t *testing.T) {
	dim := 6
	idx := newIndex(t, dim, 3, 2, 256, 10)

	makeVec := func(id int) []float32 {
		return []float32{
			float32(id),
			float32(id + 1),
			float32(id + 2),
			float32(id + 3),
			float32(id + 4),
			float32(id + 5),
		}
	}
	for i := range 2000 {
		if err := idx.Add(i, makeVec(i)); err != nil {
			t.Fatalf("Add failed for id %d: %v", i, err)
		}
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		var wg sync.WaitGroup
		for i := range 200 {
			wg.Add(2)
			go func() {
				defer wg.Done()
				var buf bytes.Buffer
				if err := idx.Save(&buf); err != nil {
					t.Errorf("Save failed: %v", err)
				}
			}()
			go func(id int) {
				defer wg.Done()
				if err := idx.Add(id, makeVec(id)); err != nil {
					t.Errorf("Add failed for id %d: %v", id, err)
				}
			}(10000 + i)
		}
		wg.Wait()
	}()

	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("concurrent Save and Add did not finish; the index deadlocked")
	}
}

func TestPQIVF_ConcurrentUpdateSearch(t *testing.T) {
	t.Setenv("HANN_SEED", "42")
	dim := 6
	idx := newIndex(t, dim, 3, 2, 256, 10)

	makeVec := func(id int) []float32 {
		return []float32{
			float32(id),
			float32(id + 1),
			float32(id + 2),
			float32(id + 3),
			float32(id + 4),
			float32(id + 5),
		}
	}
	for i := range 100 {
		if err := idx.Add(i, makeVec(i)); err != nil {
			t.Fatalf("Add failed for id %d: %v", i, err)
		}
	}
	if err := idx.Train(); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	query := makeVec(50)
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(2)
		go func(id int) {
			defer wg.Done()
			if err := idx.Update(id, makeVec(id+1000)); err != nil {
				t.Errorf("Update failed for id %d: %v", id, err)
			}
		}(i)
		go func() {
			defer wg.Done()
			// The index becomes untrained once an update lands, so an
			// error is acceptable here. The race detector is the check.
			_, _ = idx.Search(query, 1)
		}()
	}
	wg.Wait()
}

func TestPQIVF_AtomicUpdateVsAdd(t *testing.T) {
	dim := 4
	idx := newIndex(t, dim, 2, 2, 4, 5)

	if err := idx.Add(7, []float32{1, 1, 1, 1}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	const iterations = 500
	var wg sync.WaitGroup
	wg.Add(2)

	// Goroutine A updates id 7 in a loop. An update of a live id must
	// never fail, even while another goroutine races on the same id.
	go func() {
		defer wg.Done()
		for i := range iterations {
			if err := idx.Update(7, []float32{float32(i), 2, 3, 4}); err != nil {
				t.Errorf("Update(7) failed on iteration %d: %v", i, err)
				return
			}
		}
	}()

	// Goroutine B tries to add id 7 in a loop. The id is always present,
	// so every Add must fail with an "already exists" error.
	go func() {
		defer wg.Done()
		for i := range iterations {
			if err := idx.Add(7, []float32{9, 9, 9, 9}); err == nil {
				t.Errorf("Add(7) succeeded on iteration %d, id 7 should always be present", i)
				return
			}
		}
	}()

	wg.Wait()

	if stats := idx.Stats(); stats.Count != 1 {
		t.Fatalf("expected count 1 after racing Update and Add, got %d", stats.Count)
	}
}

func TestPQIVF_ConcurrentStress(t *testing.T) {
	testutil.RunConcurrentOps(t, pqivfFactory(t), 16, 8, 300)
}

package testutil

import (
	"io"
	"math/rand"
	"sync"
	"testing"
	"time"
)

// RunConcurrentOps exercises an index from many goroutines at once under the
// race detector. Each worker owns a disjoint id range. Operations therefore
// never conflict logically, and the final state is checkable, while the lock
// paths for add, update, delete, search, and save all contend. Search errors
// are ignored, because some indexes reject searches in transient states. The
// runner exists to surface data races and deadlocks, and it fails the test
// if the workers do not finish within a timeout. The final count check
// catches lost updates.
func RunConcurrentOps(t *testing.T, f Factory, dim, workers, opsPerWorker int) {
	t.Helper()
	idx := f.New()

	// Seed shared data that searches will traverse while writers run.
	const seedCount = 100
	seedData := ClusteredData(1, seedCount, dim, 4)
	for id, vec := range seedData {
		if err := idx.Add(id, CopyVector(vec)); err != nil {
			t.Fatalf("seeding Add(%d) failed: %v", id, err)
		}
	}
	if f.Train != nil {
		if err := f.Train(idx); err != nil {
			t.Fatalf("seeding Train failed: %v", err)
		}
	}

	live := make([]int, workers) // ids each worker leaves in the index
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(w + 1)))
			base := 1000 * (w + 1) // disjoint from the seed ids and other workers
			present := make(map[int]struct{})
			for op := 0; op < opsPerWorker; op++ {
				id := base + rng.Intn(32)
				_, exists := present[id]
				switch action := rng.Intn(10); {
				case action < 4 && !exists:
					vec := make([]float32, dim)
					for d := range vec {
						vec[d] = rng.Float32()*10 - 5
					}
					if err := idx.Add(id, vec); err != nil {
						errs <- err
						return
					}
					present[id] = struct{}{}
				case action < 6 && exists:
					vec := make([]float32, dim)
					for d := range vec {
						vec[d] = rng.Float32()*10 - 5
					}
					if err := idx.Update(id, vec); err != nil {
						errs <- err
						return
					}
				case action == 6 && exists:
					if err := idx.Delete(id); err != nil {
						errs <- err
						return
					}
					delete(present, id)
				case action == 7:
					// Save makes the read path contend with the writers.
					// That is the shape that deadlocked before the lock
					// fixes.
					if err := idx.Save(io.Discard); err != nil {
						errs <- err
						return
					}
				default:
					query := make([]float32, dim)
					for d := range query {
						query[d] = rng.Float32()*10 - 5
					}
					// Search errors in transient states are tolerated. Races
					// and deadlocks are what this runner is for.
					_, _ = idx.Search(query, 5)
				}
			}
			live[w] = len(present)
		}(w)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("concurrent operations did not finish within 60s, likely a deadlock")
	}
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent operation failed: %v", err)
	}

	want := seedCount
	for _, n := range live {
		want += n
	}
	if got := idx.Stats().Count; got != want {
		t.Fatalf("after concurrent operations: Stats().Count = %d, want %d", got, want)
	}
}

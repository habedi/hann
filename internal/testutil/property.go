package testutil

import (
	"bytes"
	"math/rand"
	"sort"
	"testing"

	"github.com/habedi/hann/core"
)

// Factory describes how a test constructs and prepares an index, along with
// the guarantees its Search results carry.
type Factory struct {
	// New returns an empty index.
	New func() core.Index
	// Train prepares the index for searching after data changes. Nil when the
	// index needs no training.
	Train func(core.Index) error
	// MinTrainSize is the smallest number of stored vectors Train accepts.
	// Search verification is skipped below it.
	MinTrainSize int
	// ExactDistances reports whether Search returns distances computed
	// against the stored vectors rather than quantized approximations.
	ExactDistances bool
	// SortedResults reports whether Search returns neighbors in
	// non-decreasing distance order.
	SortedResults bool
	// Metric is the metric the index was built with.
	Metric core.Metric
}

// RunPropertyOps runs a random sequence of index operations against a
// brute-force model. It checks the invariants an index must keep: operations
// on missing or duplicate ids fail, Stats().Count tracks the model, Search
// never returns an id that is not in the index, distances match the stored
// vectors when they are exact, and a save and load round-trip preserves all
// of the above. The same seed runs the same sequence.
func RunPropertyOps(t *testing.T, f Factory, dim int, seed int64, ops int) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	idx := f.New()
	model := make(map[int][]float32)
	const idSpace = 64

	randomVector := func() []float32 {
		vec := make([]float32, dim)
		for d := range vec {
			vec[d] = rng.Float32()*10 - 5
		}
		return vec
	}
	freeID := func() int {
		for {
			id := rng.Intn(idSpace)
			if _, ok := model[id]; !ok {
				return id
			}
		}
	}
	liveID := func() int {
		ids := sortedIDs(model)
		return ids[rng.Intn(len(ids))]
	}
	liveIDs := func(max int) []int {
		ids := sortedIDs(model)
		rng.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
		if len(ids) > max {
			ids = ids[:max]
		}
		return ids
	}

	for op := 0; op < ops; op++ {
		switch choice := rng.Intn(10); {
		case choice < 3 || len(model) == 0: // add a new id
			if len(model) == idSpace {
				continue
			}
			id, vec := freeID(), randomVector()
			if err := idx.Add(id, CopyVector(vec)); err != nil {
				t.Fatalf("op %d: Add(%d) failed: %v", op, id, err)
			}
			model[id] = vec
		case choice == 3: // add a duplicate id
			id := liveID()
			if err := idx.Add(id, randomVector()); err == nil {
				t.Fatalf("op %d: Add(%d) of an existing id succeeded", op, id)
			}
		case choice == 4: // delete a live id
			id := liveID()
			if err := idx.Delete(id); err != nil {
				t.Fatalf("op %d: Delete(%d) failed: %v", op, id, err)
			}
			delete(model, id)
		case choice == 5: // delete a missing id
			if len(model) == idSpace {
				continue
			}
			id := freeID()
			if err := idx.Delete(id); err == nil {
				t.Fatalf("op %d: Delete(%d) of a missing id succeeded", op, id)
			}
		case choice == 6: // update a live id
			id, vec := liveID(), randomVector()
			if err := idx.Update(id, CopyVector(vec)); err != nil {
				t.Fatalf("op %d: Update(%d) failed: %v", op, id, err)
			}
			model[id] = vec
		case choice == 7: // bulk ops on fresh and live ids
			switch rng.Intn(3) {
			case 0:
				batch := make(map[int][]float32)
				for len(batch) < 4 && len(model)+len(batch) < idSpace {
					id := rng.Intn(idSpace)
					_, inModel := model[id]
					_, inBatch := batch[id]
					if !inModel && !inBatch {
						batch[id] = randomVector()
					}
				}
				if len(batch) == 0 {
					continue
				}
				arg := make(map[int][]float32, len(batch))
				for id, vec := range batch {
					arg[id] = CopyVector(vec)
				}
				if err := core.BulkAdd(idx, arg); err != nil {
					t.Fatalf("op %d: BulkAdd failed: %v", op, err)
				}
				for id, vec := range batch {
					model[id] = vec
				}
			case 1:
				ids := liveIDs(4)
				if len(ids) == 0 {
					continue
				}
				if err := core.BulkDelete(idx, ids); err != nil {
					t.Fatalf("op %d: BulkDelete failed: %v", op, err)
				}
				for _, id := range ids {
					delete(model, id)
				}
			case 2:
				ids := liveIDs(4)
				if len(ids) == 0 {
					continue
				}
				batch := make(map[int][]float32, len(ids))
				arg := make(map[int][]float32, len(ids))
				for _, id := range ids {
					vec := randomVector()
					batch[id] = vec
					arg[id] = CopyVector(vec)
				}
				if err := core.BulkUpdate(idx, arg); err != nil {
					t.Fatalf("op %d: BulkUpdate failed: %v", op, err)
				}
				for id, vec := range batch {
					model[id] = vec
				}
			}
		case choice == 8: // save and load round-trip, then continue on the loaded index
			var buf bytes.Buffer
			if err := idx.Save(&buf); err != nil {
				t.Fatalf("op %d: Save failed: %v", op, err)
			}
			loaded := f.New()
			if err := loaded.Load(&buf); err != nil {
				t.Fatalf("op %d: Load failed: %v", op, err)
			}
			idx = loaded
		case choice == 9: // failed update must leave the index unchanged
			if len(model) == idSpace {
				continue
			}
			id := freeID()
			if err := idx.Update(id, randomVector()); err == nil {
				t.Fatalf("op %d: Update(%d) of a missing id succeeded", op, id)
			}
		}

		if op%15 == 14 || op == ops-1 {
			verifyAgainstModel(t, f, idx, model, rng, op)
		}
	}
}

// verifyAgainstModel checks the index invariants against the model after a
// batch of operations.
func verifyAgainstModel(t *testing.T, f Factory, idx core.Index, model map[int][]float32, rng *rand.Rand, op int) {
	t.Helper()
	if got, want := idx.Stats().Count, len(model); got != want {
		t.Fatalf("after op %d: Stats().Count = %d, model has %d entries", op, got, want)
	}
	if len(model) == 0 || len(model) < f.MinTrainSize {
		return
	}
	if f.Train != nil {
		if err := f.Train(idx); err != nil {
			t.Fatalf("after op %d: Train failed: %v", op, err)
		}
	}
	ids := sortedIDs(model)
	probes := 3
	if probes > len(ids) {
		probes = len(ids)
	}
	for p := 0; p < probes; p++ {
		id := ids[rng.Intn(len(ids))]
		query := CopyVector(model[id])
		k := len(model)
		if k > 10 {
			k = 10
		}
		results, err := idx.Search(query, k)
		if err != nil {
			t.Fatalf("after op %d: Search failed: %v", op, err)
		}
		prev := 0.0
		for i, n := range results {
			vec, ok := model[n.ID]
			if !ok {
				t.Fatalf("after op %d: Search returned id %d, which is not in the index", op, n.ID)
			}
			if f.ExactDistances {
				want, err := f.Metric.Distance(query, vec)
				if err != nil {
					t.Fatalf("after op %d: distance check failed: %v", op, err)
				}
				if !almostEqual(n.Distance, want) {
					t.Fatalf("after op %d: Search distance for id %d is %v, stored vector gives %v",
						op, n.ID, n.Distance, want)
				}
			}
			if f.SortedResults && i > 0 && n.Distance < prev {
				t.Fatalf("after op %d: Search results are not sorted: %v after %v", op, n.Distance, prev)
			}
			prev = n.Distance
		}
	}
}

// sortedIDs returns the model's ids in ascending order. Id selection is then
// deterministic for a given rng state.
func sortedIDs(model map[int][]float32) []int {
	ids := make([]int, 0, len(model))
	for id := range model {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}

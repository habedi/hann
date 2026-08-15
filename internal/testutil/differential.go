package testutil

import (
	"bytes"
	"fmt"
	"sort"
	"testing"

	"github.com/habedi/hann/core"
)

// RunExactDifferential compares an index against brute force in a
// configuration where the index is provably exact: searching with k equal to
// the number of stored vectors forces a complete result, so the returned ids
// must be exactly the stored ids, in non-decreasing distance order, with each
// rank's distance matching the brute-force ranking. This is a differential
// test between the index's search path and independent arithmetic, so it
// catches distance, ordering, and bookkeeping bugs that recall thresholds
// absorb.
func RunExactDifferential(t *testing.T, f Factory, dim, n, numQueries int) {
	t.Helper()
	data := ClusteredData(21, n, dim, 8)
	idx := f.New()
	arg := make(map[int][]float32, len(data))
	for id, vec := range data {
		arg[id] = CopyVector(vec)
	}
	if err := core.BulkAdd(idx, arg); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if f.Train != nil {
		if err := f.Train(idx); err != nil {
			t.Fatalf("Train failed: %v", err)
		}
	}
	for qi, query := range Queries(22, data, numQueries) {
		results, err := idx.Search(CopyVector(query), n)
		if err != nil {
			t.Fatalf("query %d: Search failed: %v", qi, err)
		}
		wantDists, err := rankedDistances(query, data, f.Metric)
		if err != nil {
			t.Fatalf("query %d: reference distances failed: %v", qi, err)
		}
		if err := compareRanking(results, data, wantDists); err != nil {
			t.Fatalf("query %d: %v", qi, err)
		}
	}
}

// RunBulkSequentialDifferential builds one index through single-item Add and
// Delete calls and another through BulkAdd and BulkDelete from the same data,
// then compares the two: counts must agree, and when the factory reports
// exact distances, complete searches on both must return identical rankings.
// The bulk and sequential paths maintain the same state through different
// code, which is exactly where past bookkeeping bugs lived.
func RunBulkSequentialDifferential(t *testing.T, f Factory, dim, n, numQueries int) {
	t.Helper()
	data := ClusteredData(31, n, dim, 8)
	ids := make([]int, 0, len(data))
	for id := range data {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	sequential := f.New()
	for _, id := range ids {
		if err := sequential.Add(id, CopyVector(data[id])); err != nil {
			t.Fatalf("Add(%d) failed: %v", id, err)
		}
	}
	bulk := f.New()
	arg := make(map[int][]float32, len(data))
	for id, vec := range data {
		arg[id] = CopyVector(vec)
	}
	if err := core.BulkAdd(bulk, arg); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}

	// Remove every third id, sequentially on one index and in bulk on the
	// other, so the delete bookkeeping of both paths is compared too.
	var removed []int
	for i := 0; i < len(ids); i += 3 {
		removed = append(removed, ids[i])
	}
	for _, id := range removed {
		if err := sequential.Delete(id); err != nil {
			t.Fatalf("Delete(%d) failed: %v", id, err)
		}
	}
	if err := core.BulkDelete(bulk, removed); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}
	for _, id := range removed {
		delete(data, id)
	}

	if f.Train != nil {
		if err := f.Train(sequential); err != nil {
			t.Fatalf("Train on the sequential index failed: %v", err)
		}
		if err := f.Train(bulk); err != nil {
			t.Fatalf("Train on the bulk index failed: %v", err)
		}
	}
	if s, b := sequential.Stats().Count, bulk.Stats().Count; s != b || s != len(data) {
		t.Fatalf("counts diverged: sequential %d, bulk %d, want %d", s, b, len(data))
	}

	for qi, query := range Queries(32, data, numQueries) {
		fromSeq, err := sequential.Search(CopyVector(query), len(data))
		if err != nil {
			t.Fatalf("query %d: Search on the sequential index failed: %v", qi, err)
		}
		fromBulk, err := bulk.Search(CopyVector(query), len(data))
		if err != nil {
			t.Fatalf("query %d: Search on the bulk index failed: %v", qi, err)
		}
		if err := sameIDSet(fromSeq, fromBulk); err != nil {
			t.Fatalf("query %d: %v", qi, err)
		}
		if !f.ExactDistances {
			continue
		}
		wantDists, err := rankedDistances(query, data, f.Metric)
		if err != nil {
			t.Fatalf("query %d: reference distances failed: %v", qi, err)
		}
		if err := compareRanking(fromSeq, data, wantDists); err != nil {
			t.Fatalf("query %d: sequential index: %v", qi, err)
		}
		if err := compareRanking(fromBulk, data, wantDists); err != nil {
			t.Fatalf("query %d: bulk index: %v", qi, err)
		}
	}
}

// RunUpdateDifferential builds one index with an initial dataset and moves
// every vector to a second dataset through Update and BulkUpdate, half each,
// then compares it against an index built directly from the second dataset.
// Searching with k above the index size must return each id exactly once on
// both, which catches stale or duplicated entries the update path left
// behind, and when the factory reports exact distances, the ranking of the
// updated index must match the reference distances of the second dataset,
// which catches an update that changed the bookkeeping but not the vector.
func RunUpdateDifferential(t *testing.T, f Factory, dim, n, numQueries int) {
	t.Helper()
	before := ClusteredData(41, n, dim, 8)
	after := ClusteredData(51, n, dim, 8)

	updated := f.New()
	arg := make(map[int][]float32, len(before))
	for id, vec := range before {
		arg[id] = CopyVector(vec)
	}
	if err := core.BulkAdd(updated, arg); err != nil {
		t.Fatalf("BulkAdd of the initial dataset failed: %v", err)
	}
	if f.Train != nil {
		if err := f.Train(updated); err != nil {
			t.Fatalf("Train before the updates failed: %v", err)
		}
	}
	ids := make([]int, 0, len(before))
	for id := range before {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	bulkArg := make(map[int][]float32, len(ids)/2)
	for i, id := range ids {
		if i%2 == 0 {
			if err := updated.Update(id, CopyVector(after[id])); err != nil {
				t.Fatalf("Update(%d) failed: %v", id, err)
			}
		} else {
			bulkArg[id] = CopyVector(after[id])
		}
	}
	if err := core.BulkUpdate(updated, bulkArg); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}

	rebuilt := f.New()
	arg = make(map[int][]float32, len(after))
	for id, vec := range after {
		arg[id] = CopyVector(vec)
	}
	if err := core.BulkAdd(rebuilt, arg); err != nil {
		t.Fatalf("BulkAdd of the second dataset failed: %v", err)
	}
	if f.Train != nil {
		if err := f.Train(rebuilt); err != nil {
			t.Fatalf("Train on the rebuilt index failed: %v", err)
		}
	}
	if u, r := updated.Stats().Count, rebuilt.Stats().Count; u != r || u != n {
		t.Fatalf("counts diverged: updated %d, rebuilt %d, want %d", u, r, n)
	}

	for qi, query := range Queries(52, after, numQueries) {
		fromUpdated, err := updated.Search(CopyVector(query), n+5)
		if err != nil {
			t.Fatalf("query %d: Search on the updated index failed: %v", qi, err)
		}
		fromRebuilt, err := rebuilt.Search(CopyVector(query), n+5)
		if err != nil {
			t.Fatalf("query %d: Search on the rebuilt index failed: %v", qi, err)
		}
		if err := coversExactly(fromUpdated, after); err != nil {
			t.Fatalf("query %d: updated index: %v", qi, err)
		}
		if err := coversExactly(fromRebuilt, after); err != nil {
			t.Fatalf("query %d: rebuilt index: %v", qi, err)
		}
		if !f.ExactDistances {
			continue
		}
		wantDists, err := rankedDistances(query, after, f.Metric)
		if err != nil {
			t.Fatalf("query %d: reference distances failed: %v", qi, err)
		}
		if err := compareRanking(fromUpdated, after, wantDists); err != nil {
			t.Fatalf("query %d: updated index: %v", qi, err)
		}
	}
}

// RunSaveLoadDifferential saves an index that has seen adds and deletes,
// loads the bytes into a fresh index, and compares complete searches on the
// two: both must return each surviving id exactly once, the distance at
// every rank must agree between original and loaded, and when the factory
// reports exact distances, both rankings must match the reference distances.
// The comparison is by rank distance rather than by id, so equidistant
// points cannot cause a spurious failure.
func RunSaveLoadDifferential(t *testing.T, f Factory, dim, n, numQueries int) {
	t.Helper()
	data := ClusteredData(61, n, dim, 8)
	original := f.New()
	arg := make(map[int][]float32, len(data))
	for id, vec := range data {
		arg[id] = CopyVector(vec)
	}
	if err := core.BulkAdd(original, arg); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	// Delete every fifth id before saving, so the serialized state includes
	// the effect of deletions, not only pristine inserts.
	for id := 0; id < n; id += 5 {
		if err := original.Delete(id); err != nil {
			t.Fatalf("Delete(%d) failed: %v", id, err)
		}
		delete(data, id)
	}
	if f.Train != nil {
		if err := f.Train(original); err != nil {
			t.Fatalf("Train failed: %v", err)
		}
	}

	var buf bytes.Buffer
	if err := original.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	loaded := f.New()
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if o, l := original.Stats().Count, loaded.Stats().Count; o != l || o != len(data) {
		t.Fatalf("counts diverged: original %d, loaded %d, want %d", o, l, len(data))
	}

	for qi, query := range Queries(62, data, numQueries) {
		fromOriginal, err := original.Search(CopyVector(query), len(data))
		if err != nil {
			t.Fatalf("query %d: Search on the original index failed: %v", qi, err)
		}
		fromLoaded, err := loaded.Search(CopyVector(query), len(data))
		if err != nil {
			t.Fatalf("query %d: Search on the loaded index failed: %v", qi, err)
		}
		if err := coversExactly(fromOriginal, data); err != nil {
			t.Fatalf("query %d: original index: %v", qi, err)
		}
		if err := coversExactly(fromLoaded, data); err != nil {
			t.Fatalf("query %d: loaded index: %v", qi, err)
		}
		for i := range fromOriginal {
			if i < len(fromLoaded) && !almostEqual(fromOriginal[i].Distance, fromLoaded[i].Distance) {
				t.Fatalf("query %d rank %d: distance %v on the original, %v after the round-trip",
					qi, i, fromOriginal[i].Distance, fromLoaded[i].Distance)
			}
		}
		if !f.ExactDistances {
			continue
		}
		wantDists, err := rankedDistances(query, data, f.Metric)
		if err != nil {
			t.Fatalf("query %d: reference distances failed: %v", qi, err)
		}
		if err := compareRanking(fromLoaded, data, wantDists); err != nil {
			t.Fatalf("query %d: loaded index: %v", qi, err)
		}
	}
}

// coversExactly checks that a search result contains every id of the data
// exactly once and nothing else.
func coversExactly(results []core.Neighbor, data map[int][]float32) error {
	if len(results) != len(data) {
		return fmt.Errorf("got %d results, want %d", len(results), len(data))
	}
	seen := make(map[int]struct{}, len(results))
	for i, nb := range results {
		if _, ok := data[nb.ID]; !ok {
			return fmt.Errorf("rank %d: id %d is not in the index", i, nb.ID)
		}
		if _, dup := seen[nb.ID]; dup {
			return fmt.Errorf("rank %d: id %d returned twice", i, nb.ID)
		}
		seen[nb.ID] = struct{}{}
	}
	return nil
}

// rankedDistances returns the distances from the query to every data point in
// non-decreasing order, computed independently of any index.
func rankedDistances(query []float32, data map[int][]float32, metric core.Metric) ([]float64, error) {
	out := make([]float64, 0, len(data))
	for _, vec := range data {
		d, err := metric.Distance(query, vec)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	sort.Float64s(out)
	return out, nil
}

// compareRanking checks that a complete search result covers exactly the data
// ids, is sorted, and matches the reference distances rank by rank. Distances
// are compared instead of ids, so equidistant points cannot cause a spurious
// failure.
func compareRanking(results []core.Neighbor, data map[int][]float32, wantDists []float64) error {
	if len(results) != len(data) {
		return fmt.Errorf("got %d results, want %d", len(results), len(data))
	}
	seen := make(map[int]struct{}, len(results))
	for i, nb := range results {
		if _, ok := data[nb.ID]; !ok {
			return fmt.Errorf("rank %d: id %d is not in the index", i, nb.ID)
		}
		if _, dup := seen[nb.ID]; dup {
			return fmt.Errorf("rank %d: id %d returned twice", i, nb.ID)
		}
		seen[nb.ID] = struct{}{}
		if i > 0 && nb.Distance < results[i-1].Distance {
			return fmt.Errorf("rank %d: distance %v after %v, results are not sorted",
				i, nb.Distance, results[i-1].Distance)
		}
		if !almostEqual(nb.Distance, wantDists[i]) {
			return fmt.Errorf("rank %d: distance %v, reference %v", i, nb.Distance, wantDists[i])
		}
	}
	return nil
}

// sameIDSet checks that two search results contain the same ids.
func sameIDSet(a, b []core.Neighbor) error {
	if len(a) != len(b) {
		return fmt.Errorf("result sizes diverged: %d and %d", len(a), len(b))
	}
	ids := make(map[int]struct{}, len(a))
	for _, nb := range a {
		ids[nb.ID] = struct{}{}
	}
	for _, nb := range b {
		if _, ok := ids[nb.ID]; !ok {
			return fmt.Errorf("id %d is in one result but not the other", nb.ID)
		}
	}
	return nil
}

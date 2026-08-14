package testutil

import (
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
	if err := idx.BulkAdd(arg); err != nil {
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
		wantDists, err := rankedDistances(query, data, f.Distance)
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
	if err := bulk.BulkAdd(arg); err != nil {
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
	if err := bulk.BulkDelete(removed); err != nil {
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
		wantDists, err := rankedDistances(query, data, f.Distance)
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

// rankedDistances returns the distances from the query to every data point in
// non-decreasing order, computed independently of any index.
func rankedDistances(query []float32, data map[int][]float32, dist core.DistanceFunc) ([]float64, error) {
	out := make([]float64, 0, len(data))
	for _, vec := range data {
		d, err := dist(query, vec)
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

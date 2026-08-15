package testutil

import (
	"reflect"
	"testing"

	"github.com/habedi/hann/core"
)

func TestClusteredData(t *testing.T) {
	const n, dim, clusters = 40, 8, 4
	data := ClusteredData(7, n, dim, clusters)
	if len(data) != n {
		t.Fatalf("expected %d vectors, got %d", n, len(data))
	}
	for id := 0; id < n; id++ {
		vec, ok := data[id]
		if !ok {
			t.Fatalf("expected id %d to be present", id)
		}
		if len(vec) != dim {
			t.Fatalf("id %d: expected dimension %d, got %d", id, dim, len(vec))
		}
	}

	// The same seed must return the same data.
	if again := ClusteredData(7, n, dim, clusters); !reflect.DeepEqual(data, again) {
		t.Error("expected identical data for equal seeds")
	}

	// A different seed must return different data.
	if other := ClusteredData(8, n, dim, clusters); reflect.DeepEqual(data, other) {
		t.Error("expected different data for different seeds")
	}
}

func TestQueries(t *testing.T) {
	const n, dim, q = 20, 6, 5
	data := ClusteredData(9, n, dim, 3)
	queries := Queries(10, data, q)
	if len(queries) != q {
		t.Fatalf("expected %d queries, got %d", q, len(queries))
	}
	for i, query := range queries {
		if len(query) != dim {
			t.Fatalf("query %d: expected dimension %d, got %d", i, dim, len(query))
		}
	}

	// The same seed must return the same queries.
	if again := Queries(10, data, q); !reflect.DeepEqual(queries, again) {
		t.Error("expected identical queries for equal seeds")
	}
}

func TestBruteForceKNN(t *testing.T) {
	// Distances from the query: id 1 and id 2 tie at 1, id 4 is at 2, and
	// id 3 is at 3, so the tie must be broken by id.
	data := map[int][]float32{
		1: {1, 0},
		2: {-1, 0},
		3: {0, 3},
		4: {2, 0},
	}
	query := []float32{0, 0}

	got, err := BruteForceKNN(query, data, 3, core.Euclidean)
	if err != nil {
		t.Fatalf("BruteForceKNN failed: %v", err)
	}
	if want := []int{1, 2, 4}; !reflect.DeepEqual(got, want) {
		t.Errorf("expected %v, got %v", want, got)
	}

	// A k larger than the data set returns every id.
	got, err = BruteForceKNN(query, data, 10, core.Euclidean)
	if err != nil {
		t.Fatalf("BruteForceKNN failed: %v", err)
	}
	if want := []int{1, 2, 4, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("expected %v, got %v", want, got)
	}

	// A query with a mismatched dimension fails.
	if _, err := BruteForceKNN([]float32{0, 0, 0}, data, 2, core.Euclidean); err == nil {
		t.Error("expected error for a mismatched query dimension, got none")
	}
}

func TestRecall(t *testing.T) {
	got := []core.Neighbor{{ID: 1}, {ID: 2}, {ID: 3}}
	if r := Recall(got, []int{1, 2, 4}); r != 2.0/3.0 {
		t.Errorf("expected recall 2/3, got %v", r)
	}
	if r := Recall(got, []int{1, 2, 3}); r != 1 {
		t.Errorf("expected recall 1, got %v", r)
	}
	if r := Recall(got, []int{7}); r != 0 {
		t.Errorf("expected recall 0 for no hits, got %v", r)
	}
	if r := Recall(got, nil); r != 0 {
		t.Errorf("expected recall 0 for an empty want set, got %v", r)
	}
}

func TestCopyVector(t *testing.T) {
	original := []float32{1, 2, 3}
	copied := CopyVector(original)
	if !reflect.DeepEqual(original, copied) {
		t.Fatalf("expected equal contents, got %v and %v", original, copied)
	}
	copied[0] = 99
	if original[0] != 1 {
		t.Error("expected the copy to have its own backing array")
	}
}

func TestAlmostEqual(t *testing.T) {
	cases := []struct {
		name string
		a, b float64
		want bool
	}{
		{"absolute tolerance accepts a tiny difference", 0, 5e-5, true},
		{"absolute tolerance rejects past its bound at small scale", 0, 1e-3, false},
		{"relative tolerance accepts at large scale", 1000, 1000.5, true},
		{"relative tolerance rejects at large scale", 1000, 1002, false},
		{"exact values agree", 3.25, 3.25, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := almostEqual(tc.a, tc.b); got != tc.want {
				t.Errorf("almostEqual(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

func TestRankedDistances(t *testing.T) {
	data := map[int][]float32{
		1: {3, 0},
		2: {1, 0},
		3: {2, 0},
	}
	got, err := rankedDistances([]float32{0, 0}, data, core.Euclidean)
	if err != nil {
		t.Fatalf("rankedDistances failed: %v", err)
	}
	if want := []float64{1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("expected %v, got %v", want, got)
	}

	// A query with a mismatched dimension fails.
	if _, err := rankedDistances([]float32{0}, data, core.Euclidean); err == nil {
		t.Error("expected error for a mismatched query dimension, got none")
	}
}

func TestCompareRanking(t *testing.T) {
	data := map[int][]float32{1: {1, 0}, 2: {2, 0}}
	wantDists := []float64{1, 2}

	ok := []core.Neighbor{{ID: 1, Distance: 1}, {ID: 2, Distance: 2}}
	if err := compareRanking(ok, data, wantDists); err != nil {
		t.Errorf("expected a matching ranking to pass, got %v", err)
	}

	cases := []struct {
		name    string
		results []core.Neighbor
		dists   []float64
	}{
		{"size mismatch", []core.Neighbor{{ID: 1, Distance: 1}}, wantDists},
		{"foreign id", []core.Neighbor{{ID: 1, Distance: 1}, {ID: 9, Distance: 2}}, wantDists},
		{"duplicate id", []core.Neighbor{{ID: 1, Distance: 1}, {ID: 1, Distance: 2}}, wantDists},
		// The distances agree with the reference within the tolerance, so the
		// ordering check is the one that fails.
		{"unsorted distances", []core.Neighbor{{ID: 1, Distance: 1}, {ID: 2, Distance: 0.99995}}, []float64{1, 1}},
		{"distance mismatch", []core.Neighbor{{ID: 1, Distance: 1}, {ID: 2, Distance: 5}}, wantDists},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := compareRanking(tc.results, data, tc.dists); err == nil {
				t.Errorf("expected an error for %s, got none", tc.name)
			}
		})
	}
}

func TestSameIDSet(t *testing.T) {
	a := []core.Neighbor{{ID: 1}, {ID: 2}}
	b := []core.Neighbor{{ID: 2}, {ID: 1}}
	if err := sameIDSet(a, b); err != nil {
		t.Errorf("expected equal id sets to pass, got %v", err)
	}
	if err := sameIDSet(a, []core.Neighbor{{ID: 1}}); err == nil {
		t.Error("expected an error for a size mismatch, got none")
	}
	if err := sameIDSet(a, []core.Neighbor{{ID: 1}, {ID: 9}}); err == nil {
		t.Error("expected an error for a foreign id, got none")
	}
}

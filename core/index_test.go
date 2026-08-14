package core

import (
	"fmt"
	"io"
	"testing"
)

// fakeIndex implements Index but not BulkIndex, so the bulk helpers must use
// their per-item fallback loops. It records the order of applied operations.
type fakeIndex struct {
	vectors map[int][]float32
	applied []int
}

func newFakeIndex() *fakeIndex {
	return &fakeIndex{vectors: make(map[int][]float32)}
}

func (f *fakeIndex) Add(id int, vector []float32) error {
	if _, exists := f.vectors[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}
	f.vectors[id] = vector
	f.applied = append(f.applied, id)
	return nil
}

func (f *fakeIndex) Delete(id int) error {
	if _, exists := f.vectors[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	delete(f.vectors, id)
	f.applied = append(f.applied, id)
	return nil
}

func (f *fakeIndex) Update(id int, vector []float32) error {
	if _, exists := f.vectors[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	f.vectors[id] = vector
	f.applied = append(f.applied, id)
	return nil
}

func (f *fakeIndex) Search(query []float32, k int) ([]Neighbor, error) { return nil, nil }
func (f *fakeIndex) Stats() IndexStats                                 { return IndexStats{Count: len(f.vectors)} }
func (f *fakeIndex) Save(w io.Writer) error                            { return nil }
func (f *fakeIndex) Load(r io.Reader) error                            { return nil }

// fakeBulkIndex adds bulk methods, so the helpers must dispatch to them
// instead of looping.
type fakeBulkIndex struct {
	fakeIndex
	bulkCalls int
}

func (f *fakeBulkIndex) BulkAdd(vectors map[int][]float32) error {
	f.bulkCalls++
	for id, vec := range vectors {
		f.vectors[id] = vec
	}
	return nil
}

func (f *fakeBulkIndex) BulkDelete(ids []int) error {
	f.bulkCalls++
	for _, id := range ids {
		delete(f.vectors, id)
	}
	return nil
}

func (f *fakeBulkIndex) BulkUpdate(updates map[int][]float32) error {
	f.bulkCalls++
	for id, vec := range updates {
		f.vectors[id] = vec
	}
	return nil
}

func TestBulkHelpersFallBackToPerItemCalls(t *testing.T) {
	idx := newFakeIndex()
	batch := map[int][]float32{3: {3}, 1: {1}, 2: {2}}
	if err := BulkAdd(idx, batch); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	// The fallback applies items in ascending id order.
	if len(idx.applied) != 3 || idx.applied[0] != 1 || idx.applied[1] != 2 || idx.applied[2] != 3 {
		t.Fatalf("expected ascending id order, got %v", idx.applied)
	}
	if err := BulkUpdate(idx, map[int][]float32{2: {20}, 1: {10}}); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}
	if got := idx.vectors[2][0]; got != 20 {
		t.Fatalf("expected updated vector for id 2, got %v", got)
	}
	if err := BulkDelete(idx, []int{1, 3}); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}
	if idx.Stats().Count != 1 {
		t.Fatalf("expected 1 entry after deletes, got %d", idx.Stats().Count)
	}

	// Errors from the per-item calls surface through the helpers.
	if err := BulkAdd(idx, map[int][]float32{2: {2}}); err == nil {
		t.Fatal("expected BulkAdd of an existing id to fail")
	}
	if err := BulkDelete(idx, []int{99}); err == nil {
		t.Fatal("expected BulkDelete of a missing id to fail")
	}
	if err := BulkUpdate(idx, map[int][]float32{99: {9}}); err == nil {
		t.Fatal("expected BulkUpdate of a missing id to fail")
	}
}

func TestBulkHelpersDispatchToBulkIndex(t *testing.T) {
	idx := &fakeBulkIndex{fakeIndex: *newFakeIndex()}
	if err := BulkAdd(idx, map[int][]float32{1: {1}, 2: {2}}); err != nil {
		t.Fatalf("BulkAdd failed: %v", err)
	}
	if err := BulkUpdate(idx, map[int][]float32{1: {10}}); err != nil {
		t.Fatalf("BulkUpdate failed: %v", err)
	}
	if err := BulkDelete(idx, []int{2}); err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}
	if idx.bulkCalls != 3 {
		t.Fatalf("expected 3 bulk fast-path calls, got %d", idx.bulkCalls)
	}
	if len(idx.applied) != 0 {
		t.Fatalf("expected no per-item calls, got %v", idx.applied)
	}
}

func TestMetricAccessorsAndRegistry(t *testing.T) {
	if Euclidean.Name() != "euclidean" || Euclidean.Normalizes() {
		t.Fatalf("unexpected euclidean metric: %q normalizes=%v", Euclidean.Name(), Euclidean.Normalizes())
	}
	if !Cosine.Normalizes() {
		t.Fatal("cosine metric must require normalization")
	}
	if (Metric{}).IsZero() != true || Euclidean.IsZero() {
		t.Fatal("IsZero must be true only for the zero metric")
	}

	custom := NewMetric("test_custom", Euclidean.Func(), false)
	if custom.IsZero() || custom.Name() != "test_custom" {
		t.Fatalf("unexpected custom metric: %+v", custom)
	}
	d, err := custom.Distance([]float32{0, 0}, []float32{3, 4})
	if err != nil || d != 5 {
		t.Fatalf("custom metric distance = %v, %v; want 5, nil", d, err)
	}

	if _, ok := MetricByName("test_custom"); ok {
		t.Fatal("unregistered metric must not resolve by name")
	}
	if err := RegisterMetric(custom); err != nil {
		t.Fatalf("RegisterMetric failed: %v", err)
	}
	got, ok := MetricByName("test_custom")
	if !ok || got.Name() != "test_custom" {
		t.Fatalf("registered metric did not resolve: %v, %v", got, ok)
	}

	// Error cases: duplicate name, empty name, and missing function.
	if err := RegisterMetric(custom); err == nil {
		t.Fatal("expected duplicate registration to fail")
	}
	if err := RegisterMetric(NewMetric("", Euclidean.Func(), false)); err == nil {
		t.Fatal("expected registration without a name to fail")
	}
	if err := RegisterMetric(NewMetric("no_func", nil, false)); err == nil {
		t.Fatal("expected registration without a function to fail")
	}

	// The built-ins resolve by their stable names.
	for _, name := range []string{"euclidean", "squared_euclidean", "manhattan", "cosine"} {
		if _, ok := MetricByName(name); !ok {
			t.Fatalf("built-in metric %q did not resolve", name)
		}
	}
}

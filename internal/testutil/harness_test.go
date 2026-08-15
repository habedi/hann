package testutil

import (
	"encoding/gob"
	"fmt"
	"io"
	"sort"
	"sync"
	"testing"

	"github.com/habedi/hann/core"
)

// scanIndex is a minimal exact index used to test the harnesses themselves:
// it stores vectors in a map and searches by a full scan. It is deliberately
// simple, so a harness failure against it points at the harness.
type scanIndex struct {
	mu      sync.Mutex
	dim     int
	vectors map[int][]float32
}

func newScanIndex(dim int) *scanIndex {
	return &scanIndex{dim: dim, vectors: make(map[int][]float32)}
}

func (s *scanIndex) Add(id int, vector []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(vector) != s.dim {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), s.dim)
	}
	if _, exists := s.vectors[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}
	s.vectors[id] = CopyVector(vector)
	return nil
}

func (s *scanIndex) Delete(id int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.vectors[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	delete(s.vectors, id)
	return nil
}

func (s *scanIndex) Update(id int, vector []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(vector) != s.dim {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), s.dim)
	}
	if _, exists := s.vectors[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	s.vectors[id] = CopyVector(vector)
	return nil
}

func (s *scanIndex) Search(query []float32, k int) ([]core.Neighbor, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}
	if len(query) != s.dim {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), s.dim)
	}
	neighbors := make([]core.Neighbor, 0, len(s.vectors))
	for id, vec := range s.vectors {
		d, err := core.Euclidean.Distance(query, vec)
		if err != nil {
			return nil, err
		}
		neighbors = append(neighbors, core.Neighbor{ID: id, Distance: d})
	}
	sort.Slice(neighbors, func(i, j int) bool {
		if neighbors[i].Distance != neighbors[j].Distance {
			return neighbors[i].Distance < neighbors[j].Distance
		}
		return neighbors[i].ID < neighbors[j].ID
	})
	if k > len(neighbors) {
		k = len(neighbors)
	}
	return neighbors[:k], nil
}

func (s *scanIndex) Stats() core.IndexStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return core.IndexStats{Count: len(s.vectors), Dimension: s.dim, Distance: core.Euclidean.Name()}
}

// scanSerialized is the gob form of scanIndex.
type scanSerialized struct {
	Dim     int
	Vectors map[int][]float32
}

func (s *scanIndex) Save(w io.Writer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return gob.NewEncoder(w).Encode(scanSerialized{Dim: s.dim, Vectors: s.vectors})
}

func (s *scanIndex) Load(r io.Reader) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var ser scanSerialized
	if err := gob.NewDecoder(r).Decode(&ser); err != nil {
		return err
	}
	s.dim = ser.Dim
	s.vectors = ser.Vectors
	return nil
}

func scanFactory(dim int) Factory {
	return Factory{
		New:            func() core.Index { return newScanIndex(dim) },
		ExactDistances: true,
		SortedResults:  true,
		Metric:         core.Euclidean,
	}
}

// The tests below run every harness against the exact scan index, so the
// harnesses themselves are exercised by this package's own test suite. A
// failure here means the harness rejects a correct index.

func TestRunExactDifferentialHarness(t *testing.T) {
	RunExactDifferential(t, scanFactory(8), 8, 120, 5)
}

func TestRunBulkSequentialDifferentialHarness(t *testing.T) {
	RunBulkSequentialDifferential(t, scanFactory(8), 8, 120, 5)
}

func TestRunUpdateDifferentialHarness(t *testing.T) {
	RunUpdateDifferential(t, scanFactory(8), 8, 120, 5)
}

func TestRunSaveLoadDifferentialHarness(t *testing.T) {
	RunSaveLoadDifferential(t, scanFactory(8), 8, 120, 5)
}

func TestRunPropertyOpsHarness(t *testing.T) {
	RunPropertyOps(t, scanFactory(8), 8, 3, 300)
}

func TestRunConcurrentOpsHarness(t *testing.T) {
	RunConcurrentOps(t, scanFactory(8), 8, 4, 60)
}

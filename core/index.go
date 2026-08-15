// Package core declares the Index interface the hann indexes implement,
// together with the distance functions, the metric registry, and the
// shared vector operations.
package core

import (
	"io"
	"sort"
)

// Index is the interface every index in Hann implements. Implementations
// must be safe for concurrent use: any method may be called from multiple
// goroutines at once.
type Index interface {

	// Add inserts a vector with a given id into the index.
	// It returns an error when the id already exists or the vector has the
	// wrong dimension.
	// The index stores the slice without copying it, and a metric that
	// normalizes does so in place, so the caller must not modify the slice
	// after the call. This applies to every method that takes a vector,
	// except Search, which copies its query.
	Add(id int, vector []float32) error

	// Delete removes the vector with the given id from the index.
	// It returns an error when the id does not exist.
	Delete(id int) error

	// Update replaces the vector associated with the given id atomically.
	// It returns an error when the id does not exist or the vector has the
	// wrong dimension. A failed update leaves the index unchanged.
	Update(id int, vector []float32) error

	// Search returns the ids and distances of the k nearest neighbors for a
	// query vector, in non-decreasing distance order.
	Search(query []float32, k int) ([]Neighbor, error)

	// Stats returns metadata about the index, such as count and dimension.
	Stats() IndexStats

	// Save persists the index state to the given writer.
	Save(w io.Writer) error

	// Load initializes the index from a previously saved state.
	Load(r io.Reader) error
}

// BulkIndex is implemented by indexes that apply batches under a single
// critical section. Each batched method behaves like its single-item
// counterpart applied to every element.
type BulkIndex interface {
	Index

	// BulkAdd inserts multiple vectors into the index.
	BulkAdd(vectors map[int][]float32) error

	// BulkDelete removes multiple vectors from the index.
	BulkDelete(ids []int) error

	// BulkUpdate replaces multiple vectors in the index.
	BulkUpdate(updates map[int][]float32) error
}

// Trainer is implemented by indexes that require a training step before they
// can be searched, such as the PQIVF index.
type Trainer interface {
	// Train builds the quantization state from the stored vectors.
	Train() error
}

// BulkAdd inserts multiple vectors. It uses the index's batched fast path
// when it has one. Otherwise it makes per-item calls in ascending id order.
func BulkAdd(idx Index, vectors map[int][]float32) error {
	if bulk, ok := idx.(BulkIndex); ok {
		return bulk.BulkAdd(vectors)
	}
	for _, id := range sortedKeys(vectors) {
		if err := idx.Add(id, vectors[id]); err != nil {
			return err
		}
	}
	return nil
}

// BulkDelete removes multiple vectors. It uses the index's batched fast path
// when it has one. Otherwise it makes per-item calls.
func BulkDelete(idx Index, ids []int) error {
	if bulk, ok := idx.(BulkIndex); ok {
		return bulk.BulkDelete(ids)
	}
	for _, id := range ids {
		if err := idx.Delete(id); err != nil {
			return err
		}
	}
	return nil
}

// BulkUpdate replaces multiple vectors. It uses the index's batched fast path
// when it has one. Otherwise it makes per-item calls in ascending id order.
func BulkUpdate(idx Index, updates map[int][]float32) error {
	if bulk, ok := idx.(BulkIndex); ok {
		return bulk.BulkUpdate(updates)
	}
	for _, id := range sortedKeys(updates) {
		if err := idx.Update(id, updates[id]); err != nil {
			return err
		}
	}
	return nil
}

// sortedKeys returns the map's keys in ascending order. This gives the
// fallback loops a fixed order, so a batch is applied the same way every run.
func sortedKeys(m map[int][]float32) []int {
	keys := make([]int, 0, len(m))
	for id := range m {
		keys = append(keys, id)
	}
	sort.Ints(keys)
	return keys
}

// Neighbor holds a neighbor's id and its computed distance.
type Neighbor struct {
	ID       int     // the identifier of the neighbor.
	Distance float64 // the computed distance to the neighbor.
}

// IndexStats contains metadata about the index.
type IndexStats struct {
	Count     int    // total number of indexed vectors.
	Dimension int    // dimensionality of vectors.
	Distance  string // name of the metric used by the index.

	// FallbackSearches is the number of searches so far that fell back to a
	// brute-force scan because the index structure found too few candidates.
	FallbackSearches int64
}

package core

// Index represents a generic interface for an approximate nearest neighbors search index.
// All indexes in Hann must implement the methods defined in this interface.
type Index interface {

	// Add inserts a vector with a given id into the index.
	// id: the identifier for the vector.
	// vector: the vector to be added.
	// Returns an error if the operation fails.
	Add(id int, vector []float32) error

	// BulkAdd inserts multiple vectors into the index.
	// vectors: a map where the key is the vector id and the value is the vector.
	// Returns an error if the operation fails.
	BulkAdd(vectors map[int][]float32) error

	// Delete removes the vector with the given id from the index.
	// id: the identifier for the vector to be removed.
	// Returns an error if the operation fails.
	Delete(id int) error

	// BulkDelete removes multiple vectors from the index.
	// ids: a slice of vector ids to be removed.
	// Returns an error if the operation fails.
	BulkDelete(ids []int) error

	// Update modifies the vector associated with the given id.
	// id: the identifier for the vector to be updated.
	// vector: the new vector.
	// Returns an error if the operation fails.
	Update(id int, vector []float32) error

	// BulkUpdate updates multiple vectors in the index.
	// updates: a map where the key is the vector id and the value is the new vector.
	// Returns an error if the operation fails.
	BulkUpdate(updates map[int][]float32) error

	// Search returns the ids and distances of the k nearest neighbors for a query vector.
	// query: the vector to search for.
	// k: the number of nearest neighbors to return.
	// Returns a slice of Neighbor structs and an error if the operation fails.
	Search(query []float32, k int) ([]Neighbor, error)

	// Stats returns metadata about the index, such as count and dimensionality.
	// Returns an IndexStats struct containing the metadata.
	Stats() IndexStats

	// Save persists the index state to the specified file.
	// path: the file path where the index state will be saved.
	// Returns an error if the operation fails.
	Save(path string) error

	// Load initializes the index from a previously saved state.
	// path: the file path from which the index state will be loaded.
	// Returns an error if the operation fails.
	Load(path string) error
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
	Size      int    // (approximate) size of the index in bytes.
	Distance  string // name of the distance function used by the index.
}

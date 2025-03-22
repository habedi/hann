package rpt

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"

	"github.com/habedi/hann/core"
)

// treeNode represents a node in the random projection tree.
type treeNode struct {
	// if leaf, points stores the IDs of vectors in this node.
	isLeaf bool
	points []int

	// if internal, store the projection vector, threshold, and child pointers.
	projection []float32
	threshold  float64
	left       *treeNode
	right      *treeNode
}

// RPTIndex implements a simple Random Projection Trees index.
// It lazily builds a tree over the stored points.
type RPTIndex struct {
	mu        sync.RWMutex
	dimension int
	points    map[int][]float32 // id -> vector
	tree      *treeNode         // cached tree built from points
	dirty     bool              // true if the tree needs rebuilding
}

const leafCapacity = 10

// NewRPTIndex creates a new RPTIndex for vectors of the given dimension.
func NewRPTIndex(dimension int) *RPTIndex {
	return &RPTIndex{
		dimension: dimension,
		points:    make(map[int][]float32),
		dirty:     true,
	}
}

// buildTreeRecursive recursively builds a treeNode from the given point IDs.
func buildTreeRecursive(ids []int, points map[int][]float32, dimension int) *treeNode {
	// If there are few points, make a leaf node.
	if len(ids) <= leafCapacity {
		return &treeNode{
			isLeaf: true,
			points: ids,
		}
	}

	// Generate a random projection vector (unit vector).
	proj := make([]float32, dimension)
	var norm float64
	for i := 0; i < dimension; i++ {
		v := rand.Float32()*2 - 1 // value in [-1,1]
		proj[i] = v
		norm += float64(v * v)
	}
	norm = math.Sqrt(norm)
	// Avoid division by zero.
	if norm < 1e-8 {
		norm = 1
	}
	for i := 0; i < dimension; i++ {
		proj[i] /= float32(norm)
	}

	// Compute dot product for each point.
	type pair struct {
		id  int
		dot float64
	}
	pairs := make([]pair, len(ids))
	for i, id := range ids {
		vec := points[id]
		var dot float64
		for j := 0; j < dimension; j++ {
			dot += float64(vec[j]) * float64(proj[j])
		}
		pairs[i] = pair{id, dot}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].dot < pairs[j].dot
	})

	mid := len(pairs) / 2
	threshold := pairs[mid].dot

	var leftIDs, rightIDs []int
	for _, p := range pairs {
		if p.dot < threshold {
			leftIDs = append(leftIDs, p.id)
		} else {
			rightIDs = append(rightIDs, p.id)
		}
	}
	// Avoid empty children by splitting evenly.
	if len(leftIDs) == 0 {
		leftIDs = rightIDs[:len(rightIDs)/2]
		rightIDs = rightIDs[len(rightIDs)/2:]
	} else if len(rightIDs) == 0 {
		rightIDs = leftIDs[len(leftIDs)/2:]
		leftIDs = leftIDs[:len(leftIDs)/2]
	}

	return &treeNode{
		isLeaf:     false,
		projection: proj,
		threshold:  threshold,
		left:       buildTreeRecursive(leftIDs, points, dimension),
		right:      buildTreeRecursive(rightIDs, points, dimension),
	}
}

// buildTree rebuilds the tree from the current points.
func (r *RPTIndex) buildTree() {
	ids := make([]int, 0, len(r.points))
	for id := range r.points {
		ids = append(ids, id)
	}
	// Shuffle to avoid worst-case splits.
	rand.Shuffle(len(ids), func(i, j int) {
		ids[i], ids[j] = ids[j], ids[i]
	})
	r.tree = buildTreeRecursive(ids, r.points, r.dimension)
	r.dirty = false
}

// searchTree traverses the tree to find candidate point IDs.
func searchTree(node *treeNode, query []float32, dimension int) []int {
	if node == nil {
		return nil
	}
	if node.isLeaf {
		return node.points
	}
	// Compute dot product for query.
	var dot float64
	for i := 0; i < dimension; i++ {
		dot += float64(query[i]) * float64(node.projection[i])
	}
	if dot < node.threshold {
		return searchTree(node.left, query, dimension)
	}
	return searchTree(node.right, query, dimension)
}

// Add inserts a vector with the given id.
func (r *RPTIndex) Add(id int, vector []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(vector) != r.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), r.dimension)
	}
	if _, exists := r.points[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}
	r.points[id] = vector
	r.dirty = true
	return nil
}

// BulkAdd inserts multiple vectors into the index.
func (r *RPTIndex) BulkAdd(vectors map[int][]float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for id, vector := range vectors {
		if len(vector) != r.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), r.dimension, id)
		}
		if _, exists := r.points[id]; exists {
			return fmt.Errorf("id %d already exists", id)
		}
		r.points[id] = vector
	}
	r.dirty = true
	return nil
}

// Delete removes the vector with the given id.
func (r *RPTIndex) Delete(id int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.points[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	delete(r.points, id)
	r.dirty = true
	return nil
}

// BulkDelete removes multiple vectors from the index.
func (r *RPTIndex) BulkDelete(ids []int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, id := range ids {
		if _, exists := r.points[id]; exists {
			delete(r.points, id)
		}
	}
	r.dirty = true
	return nil
}

// Update modifies the vector associated with the given id.
func (r *RPTIndex) Update(id int, vector []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(vector) != r.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), r.dimension)
	}
	if _, exists := r.points[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	r.points[id] = vector
	r.dirty = true
	return nil
}

// BulkUpdate updates multiple vectors in the index.
func (r *RPTIndex) BulkUpdate(updates map[int][]float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for id, vector := range updates {
		if len(vector) != r.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), r.dimension, id)
		}
		if _, exists := r.points[id]; !exists {
			return fmt.Errorf("id %d not found", id)
		}
		r.points[id] = vector
	}
	r.dirty = true
	return nil
}

// Search returns the k nearest neighbors (ids and distances) for a query vector.
func (r *RPTIndex) Search(query []float32, k int, distance core.DistanceFunc) ([]core.Neighbor, error) {
	r.mu.RLock()
	if len(query) != r.dimension {
		r.mu.RUnlock()
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), r.dimension)
	}
	if len(r.points) == 0 {
		r.mu.RUnlock()
		return nil, errors.New("index is empty")
	}
	if r.dirty {
		// Upgrade lock: release read lock, rebuild tree with write lock, then reacquire read lock.
		r.mu.RUnlock()
		r.mu.Lock()
		if r.dirty { // double-check flag
			r.buildTree()
		}
		r.mu.Unlock()
		r.mu.RLock()
	}
	candidateIDs := searchTree(r.tree, query, r.dimension)
	r.mu.RUnlock()

	// Compute distances for candidate IDs.
	var neighbors []core.Neighbor
	for _, id := range candidateIDs {
		vec := r.points[id]
		d := distance(query, vec)
		neighbors = append(neighbors, core.Neighbor{ID: id, Distance: d})
	}
	// Fallback: if not enough candidates, scan all points.
	if len(neighbors) < k {
		r.mu.RLock()
		for id, vec := range r.points {
			already := false
			for _, n := range neighbors {
				if n.ID == id {
					already = true
					break
				}
			}
			if !already {
				d := distance(query, vec)
				neighbors = append(neighbors, core.Neighbor{ID: id, Distance: d})
			}
		}
		r.mu.RUnlock()
	}
	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})
	if k > len(neighbors) {
		k = len(neighbors)
	}
	return neighbors[:k], nil
}

// Stats returns metadata about the index.
func (r *RPTIndex) Stats() core.IndexStats {
	r.mu.RLock()
	defer r.mu.RUnlock()
	count := len(r.points)
	size := count * r.dimension * 4
	return core.IndexStats{
		Count:     count,
		Dimension: r.dimension,
		Size:      size,
	}
}

// Persistence: We persist only the points and dimension; the tree is rebuilt on demand.
type rptSerialized struct {
	Dimension int
	Points    map[int][]float32
}

func (r *RPTIndex) GobEncode() ([]byte, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ser := rptSerialized{
		Dimension: r.dimension,
		Points:    r.points,
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(ser); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (r *RPTIndex) GobDecode(data []byte) error {
	var ser rptSerialized
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&ser); err != nil {
		return err
	}
	r.dimension = ser.Dimension
	r.points = ser.Points
	r.dirty = true
	return nil
}

// Save persists the index state to a file.
func (r *RPTIndex) Save(path string) error {
	r.mu.RLock()
	defer r.mu.RUnlock()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	return enc.Encode(r)
}

// Load initializes the index from a saved state.
func (r *RPTIndex) Load(path string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	return dec.Decode(r)
}

// Verify that RPTIndex implements the core.Index interface.
var _ core.Index = (*RPTIndex)(nil)

func init() {
	gob.Register(&RPTIndex{})
}

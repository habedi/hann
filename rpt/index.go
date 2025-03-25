package rpt

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"github.com/habedi/hann/core"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
)

// NewRPTIndex creates a new RPT (Random Projection Tree) index.
// It initializes parameters like dimension, leaf capacity, candidate projections, parallel threshold, and probe margin.
func NewRPTIndex(
	dimension int,
	leafCapacity int,
	candidateProjections int,
	parallelThreshold int,
	probeMargin float64,
) *RPTIndex {
	return &RPTIndex{
		dimension:            dimension,
		points:               make(map[int][]float32),
		dirty:                true, // marks that the tree needs to be rebuilt
		LeafCapacity:         leafCapacity,
		CandidateProjections: candidateProjections,
		ParallelThreshold:    parallelThreshold,
		ProbeMargin:          probeMargin,
		Distance:             core.Euclidean, // default distance function
		DistanceName:         "euclidean",
	}
}

// treeNode represents a node in the random projection tree.
// It holds the projection, threshold, and pointers to left/right children.
// If isLeaf is true, the node holds a list of point ids.
type treeNode struct {
	isLeaf     bool      // true if this node is a leaf
	points     []int     // ids of points in the leaf
	projection []float32 // projection vector used for splitting at this node
	threshold  float64   // split threshold (median value)
	left       *treeNode // left child node
	right      *treeNode // right child node
}

// RPTIndex is the main structure for the random projection tree index.
// It holds all points, the tree root, and configuration parameters.
type RPTIndex struct {
	mu                   sync.RWMutex      // protects concurrent access
	dimension            int               // dimension of each vector
	points               map[int][]float32 // mapping of point id to vector
	tree                 *treeNode         // root of the random projection tree
	dirty                bool              // indicates if the tree needs to be rebuilt
	Distance             core.DistanceFunc // function to compute distance between vectors
	DistanceName         string            // name of the distance metric
	LeafCapacity         int               // maximum number of points in a leaf
	CandidateProjections int               // number of random projections to try when splitting
	ParallelThreshold    int               // threshold to trigger parallel tree building
	ProbeMargin          float64           // margin for multi-probe search
}

// buildTreeRecursive builds the tree recursively using random projections.
// It splits the given set of point ids based on a randomly chosen projection.
func buildTreeRecursive(ids []int, points map[int][]float32, dimension int,
	distance core.DistanceFunc, rnd *rand.Rand,
	leafCapacity int, candidateProjections int, parallelThreshold int) *treeNode {

	// If the number of points is small enough, create a leaf node.
	if len(ids) <= leafCapacity {
		return &treeNode{
			isLeaf: true,
			points: ids,
		}
	}

	// Define a candidate structure to store the projection and split details.
	type candidate struct {
		proj      []float32 // random projection vector
		threshold float64   // median threshold along projection
		leftIDs   []int     // point ids going to left child
		rightIDs  []int     // point ids going to right child
		imbalance int       // difference in count between left and right sets
	}
	var bestCandidate *candidate

	// Try multiple random projections to find a good split.
	for c := 0; c < candidateProjections; c++ {
		proj := make([]float32, dimension)
		var norm float64
		// Generate a random vector.
		for i := 0; i < dimension; i++ {
			v := rnd.Float32()*2 - 1
			proj[i] = v
			norm += float64(v * v)
		}
		norm = math.Sqrt(norm)
		if norm < 1e-8 {
			norm = 1
		}
		// Normalize the projection.
		for i := 0; i < dimension; i++ {
			proj[i] /= float32(norm)
		}

		// Compute dot products of all points with the projection.
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
		// Sort points by their projection value.
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].dot < pairs[j].dot
		})
		// Choose the median as threshold.
		mid := len(pairs) / 2
		threshold := pairs[mid].dot

		// Split ids into left and right groups.
		var leftIDs, rightIDs []int
		for _, p := range pairs {
			if p.dot < threshold {
				leftIDs = append(leftIDs, p.id)
			} else {
				rightIDs = append(rightIDs, p.id)
			}
		}
		// Fallback: if one side is empty, split evenly.
		if len(leftIDs) == 0 || len(rightIDs) == 0 {
			mid = len(ids) / 2
			leftIDs = make([]int, mid)
			rightIDs = make([]int, len(ids)-mid)
			copy(leftIDs, ids[:mid])
			copy(rightIDs, ids[mid:])
		}
		imbalance := int(math.Abs(float64(len(leftIDs) - len(rightIDs))))
		cand := candidate{
			proj:      proj,
			threshold: threshold,
			leftIDs:   leftIDs,
			rightIDs:  rightIDs,
			imbalance: imbalance,
		}
		// Choose the candidate with the smallest imbalance.
		if bestCandidate == nil || cand.imbalance < bestCandidate.imbalance {
			bestCandidate = &cand
		}
	}

	var leftChild, rightChild *treeNode
	// If many points, build subtrees in parallel.
	if len(ids) > parallelThreshold {
		var wg sync.WaitGroup
		wg.Add(2)
		leftRnd := rand.New(rand.NewSource(core.GetSeed() + 1))
		rightRnd := rand.New(rand.NewSource(core.GetSeed() + 2))
		go func() {
			defer wg.Done()
			leftChild = buildTreeRecursive(bestCandidate.leftIDs, points, dimension, distance,
				leftRnd, leafCapacity, candidateProjections, parallelThreshold)
		}()
		go func() {
			defer wg.Done()
			rightChild = buildTreeRecursive(bestCandidate.rightIDs, points, dimension, distance,
				rightRnd, leafCapacity, candidateProjections, parallelThreshold)
		}()
		wg.Wait()
	} else {
		// Otherwise, build recursively in a single thread.
		leftChild = buildTreeRecursive(bestCandidate.leftIDs, points, dimension, distance, rnd,
			leafCapacity, candidateProjections, parallelThreshold)
		rightChild = buildTreeRecursive(bestCandidate.rightIDs, points, dimension, distance, rnd,
			leafCapacity, candidateProjections, parallelThreshold)
	}

	// Return an internal node with the best projection and split.
	return &treeNode{
		isLeaf:     false,
		projection: bestCandidate.proj,
		threshold:  bestCandidate.threshold,
		left:       leftChild,
		right:      rightChild,
	}
}

// buildTree constructs the random projection tree from all stored points.
func (r *RPTIndex) buildTree() {
	// Collect all point ids.
	ids := make([]int, 0, len(r.points))
	for id := range r.points {
		ids = append(ids, id)
	}
	// Shuffle the ids to avoid bias.
	rand.Shuffle(len(ids), func(i, j int) {
		ids[i], ids[j] = ids[j], ids[i]
	})
	// Use a new random source for building the tree.
	localRand := rand.New(rand.NewSource(core.GetSeed()))
	r.tree = buildTreeRecursive(ids, r.points, r.dimension, r.Distance, localRand, r.LeafCapacity,
		r.CandidateProjections, r.ParallelThreshold)
	r.dirty = false // tree is now up to date
}

// searchTreeMultiProbeWithMargin searches the tree for candidate point ids using multi-probing.
// It follows both branches if the projection value is close to the threshold (within margin).
func searchTreeMultiProbeWithMargin(node *treeNode, query []float32, dimension int,
	distance core.DistanceFunc, margin float64) []int {
	if node == nil {
		return nil
	}
	// If it's a leaf, return all point ids.
	if node.isLeaf {
		return node.points
	}
	// Compute the dot product with the node's projection.
	var dot float64
	for i := 0; i < dimension; i++ {
		dot += float64(query[i]) * float64(node.projection[i])
	}
	// If close to threshold, probe both children.
	if math.Abs(dot-node.threshold) < margin {
		leftIDs := searchTreeMultiProbeWithMargin(node.left, query, dimension, distance, margin)
		rightIDs := searchTreeMultiProbeWithMargin(node.right, query, dimension, distance, margin)
		return append(leftIDs, rightIDs...)
	} else if dot < node.threshold {
		return searchTreeMultiProbeWithMargin(node.left, query, dimension, distance, margin)
	}
	return searchTreeMultiProbeWithMargin(node.right, query, dimension, distance, margin)
}

// unionInts returns the union of two integer slices (removing duplicates).
func unionInts(a, b []int) []int {
	m := make(map[int]struct{})
	for _, x := range a {
		m[x] = struct{}{}
	}
	for _, x := range b {
		m[x] = struct{}{}
	}
	result := make([]int, 0, len(m))
	for x := range m {
		result = append(result, x)
	}
	return result
}

// computeDistances calculates the distance from the query to each point id in the list.
// It does this in parallel across available CPUs.
func (r *RPTIndex) computeDistances(query []float32, ids []int) []core.Neighbor {
	neighbors := make([]core.Neighbor, len(ids))
	numWorkers := runtime.NumCPU()
	chunkSize := (len(ids) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(ids) {
			end = len(ids)
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				id := ids[j]
				vec := r.points[id]
				d := r.Distance(query, vec)
				neighbors[j] = core.Neighbor{ID: id, Distance: d}
			}
		}(start, end)
	}
	wg.Wait()
	return neighbors
}

// Search returns the k nearest neighbors to the query vector.
// It rebuilds the tree if needed and uses multi-probe search to get candidate ids.
func (r *RPTIndex) Search(query []float32, k int) ([]core.Neighbor, error) {
	r.mu.RLock()
	if len(query) != r.dimension {
		r.mu.RUnlock()
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d",
			len(query), r.dimension)
	}
	if len(r.points) == 0 {
		r.mu.RUnlock()
		return nil, errors.New("index is empty")
	}
	// Copy the query to avoid modifying the original.
	queryCopy := make([]float32, len(query))
	copy(queryCopy, query)
	query = queryCopy

	// If the tree is dirty, rebuild it.
	if r.dirty {
		r.mu.RUnlock()
		r.mu.Lock()
		if r.dirty {
			r.buildTree()
		}
		r.mu.Unlock()
		r.mu.RLock()
	}
	// Get candidate ids using multi-probe search.
	candidateIDs := searchTreeMultiProbeWithMargin(r.tree, query, r.dimension, r.Distance, r.ProbeMargin)
	// If not enough candidates, try with a larger margin.
	if len(candidateIDs) < k*2 {
		candidateIDsAlt := searchTreeMultiProbeWithMargin(r.tree, query, r.dimension, r.Distance, r.ProbeMargin*2)
		candidateIDs = unionInts(candidateIDs, candidateIDsAlt)
	}
	r.mu.RUnlock()

	// Compute distances for candidate points.
	neighbors := r.computeDistances(query, candidateIDs)
	// If still not enough, add extra points.
	if len(neighbors) < k {
		r.mu.RLock()
		candidateSet := make(map[int]struct{}, len(candidateIDs))
		for _, id := range candidateIDs {
			candidateSet[id] = struct{}{}
		}
		var missingIDs []int
		for id := range r.points {
			if _, exists := candidateSet[id]; !exists {
				missingIDs = append(missingIDs, id)
			}
		}
		r.mu.RUnlock()
		extraNeighbors := r.computeDistances(query, missingIDs)
		neighbors = append(neighbors, extraNeighbors...)
	}
	// Sort by distance.
	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})
	if k > len(neighbors) {
		k = len(neighbors)
	}
	return neighbors[:k], nil
}

// Add inserts a new point with the given id and vector into the index.
// It marks the tree as dirty so it will be rebuilt.
func (r *RPTIndex) Add(id int, vector []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(vector) != r.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(vector), r.dimension)
	}
	if _, exists := r.points[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}
	r.points[id] = vector
	r.dirty = true
	return nil
}

// BulkAdd inserts multiple points into the index and marks the tree as dirty.
func (r *RPTIndex) BulkAdd(vectors map[int][]float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for id, vector := range vectors {
		if len(vector) != r.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d",
				len(vector), r.dimension, id)
		}
		if _, exists := r.points[id]; exists {
			return fmt.Errorf("id %d already exists", id)
		}
		r.points[id] = vector
	}
	r.dirty = true
	return nil
}

// Delete removes a point by its id and marks the tree as dirty.
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

// BulkDelete removes multiple points from the index and marks the tree as dirty.
func (r *RPTIndex) BulkDelete(ids []int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, id := range ids {
		delete(r.points, id)
	}
	r.dirty = true
	return nil
}

// Update changes the vector of an existing point and marks the tree as dirty.
func (r *RPTIndex) Update(id int, vector []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(vector) != r.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(vector), r.dimension)
	}
	if _, exists := r.points[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	r.points[id] = vector
	r.dirty = true
	return nil
}

// BulkUpdate updates multiple points in the index.
func (r *RPTIndex) BulkUpdate(updates map[int][]float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for id, vector := range updates {
		if len(vector) != r.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d",
				len(vector), r.dimension, id)
		}
		if _, exists := r.points[id]; !exists {
			return fmt.Errorf("id %d not found", id)
		}
		r.points[id] = vector
	}
	r.dirty = true
	return nil
}

// Stats returns some basic statistics about the index.
func (r *RPTIndex) Stats() core.IndexStats {
	r.mu.RLock()
	defer r.mu.RUnlock()
	count := len(r.points)
	return core.IndexStats{
		Count:     count,
		Dimension: r.dimension,
		Distance:  "euclidean",
	}
}

// rptSerialized is used to serialize the index using gob.
type rptSerialized struct {
	Dimension    int
	Points       map[int][]float32
	DistanceName string
}

// GobEncode serializes the index to bytes using gob.
func (r *RPTIndex) GobEncode() ([]byte, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ser := rptSerialized{
		Dimension:    r.dimension,
		Points:       r.points,
		DistanceName: "euclidean",
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(ser); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode deserializes the index from gob data.
func (r *RPTIndex) GobDecode(data []byte) error {
	var ser rptSerialized
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&ser); err != nil {
		return err
	}
	r.dimension = ser.Dimension
	r.points = ser.Points
	r.DistanceName = "euclidean"
	r.dirty = true // mark tree as dirty so it will be rebuilt
	return nil
}

// Save writes the serialized index to disk.
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

// Load reads the serialized index from disk.
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

// Check that RPTIndex implements the core.Index interface.
var _ core.Index = (*RPTIndex)(nil)

// Register RPTIndex for gob encoding.
func init() {
	gob.Register(&RPTIndex{})
}

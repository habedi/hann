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

// Constants for the RPT index (can be tuned).
const (
	leafCapacity         = 10  // maximum points in a leaf node
	candidateProjections = 3   // number of candidate projections tried per split
	parallelThreshold    = 100 // if number of points > threshold, build children in parallel
	probeMargin          = 0.15
)

// NewRPTIndex creates a new RPTIndex for vectors of the given dimension.
func NewRPTIndex(dimension int, distance core.DistanceFunc, distanceName string) *RPTIndex {
	return &RPTIndex{
		dimension:    dimension,
		points:       make(map[int][]float32),
		dirty:        true,
		Distance:     distance,
		DistanceName: distanceName,
	}
}

// treeNode represents a node in the random projection tree.
type treeNode struct {
	isLeaf     bool
	points     []int
	projection []float32
	threshold  float64
	left       *treeNode
	right      *treeNode
}

// RPTIndex implements a simple Random Projection Trees index.
type RPTIndex struct {
	mu        sync.RWMutex
	dimension int
	points    map[int][]float32 // id -> vector
	tree      *treeNode         // cached tree built from points
	dirty     bool              // true if the tree needs rebuilding

	Distance     core.DistanceFunc // internal distance function
	DistanceName string            // human–readable name for the distance metric
}

// buildTreeRecursive recursively builds a treeNode from the given point IDs.
func buildTreeRecursive(ids []int, points map[int][]float32, dimension int, distance core.DistanceFunc, rnd *rand.Rand) *treeNode {
	if len(ids) <= leafCapacity {
		return &treeNode{
			isLeaf: true,
			points: ids,
		}
	}

	type candidate struct {
		proj      []float32
		threshold float64
		leftIDs   []int
		rightIDs  []int
		imbalance int
	}
	var bestCandidate *candidate
	for c := 0; c < candidateProjections; c++ {
		proj := make([]float32, dimension)
		var norm float64
		for i := 0; i < dimension; i++ {
			v := rnd.Float32()*2 - 1
			proj[i] = v
			norm += float64(v * v)
		}
		norm = math.Sqrt(norm)
		if norm < 1e-8 {
			norm = 1
		}
		for i := 0; i < dimension; i++ {
			proj[i] /= float32(norm)
		}

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
		if bestCandidate == nil || cand.imbalance < bestCandidate.imbalance {
			bestCandidate = &cand
		}
	}

	var leftChild, rightChild *treeNode
	if len(ids) > parallelThreshold {
		var wg sync.WaitGroup
		wg.Add(2)
		// Create new local rand instances for each goroutine.
		leftRnd := rand.New(rand.NewSource(core.GetSeed()))
		rightRnd := rand.New(rand.NewSource(core.GetSeed()))
		go func() {
			defer wg.Done()
			leftChild = buildTreeRecursive(bestCandidate.leftIDs, points, dimension, distance, leftRnd)
		}()
		go func() {
			defer wg.Done()
			rightChild = buildTreeRecursive(bestCandidate.rightIDs, points, dimension, distance, rightRnd)
		}()
		wg.Wait()
	} else {
		leftChild = buildTreeRecursive(bestCandidate.leftIDs, points, dimension, distance, rnd)
		rightChild = buildTreeRecursive(bestCandidate.rightIDs, points, dimension, distance, rnd)
	}

	return &treeNode{
		isLeaf:     false,
		projection: bestCandidate.proj,
		threshold:  bestCandidate.threshold,
		left:       leftChild,
		right:      rightChild,
	}
}

// buildTree rebuilds the tree from the current points.
func (r *RPTIndex) buildTree() {
	ids := make([]int, 0, len(r.points))
	for id := range r.points {
		ids = append(ids, id)
	}
	rand.Shuffle(len(ids), func(i, j int) {
		ids[i], ids[j] = ids[j], ids[i]
	})
	localRand := rand.New(rand.NewSource(core.GetSeed()))
	r.tree = buildTreeRecursive(ids, r.points, r.dimension, r.Distance, localRand)
	r.dirty = false
}

// searchTreeMultiProbeWithMargin recursively traverses the tree to find candidate point IDs.
func searchTreeMultiProbeWithMargin(node *treeNode, query []float32, dimension int, distance core.DistanceFunc, margin float64) []int {
	if node == nil {
		return nil
	}
	if node.isLeaf {
		return node.points
	}
	var dot float64
	for i := 0; i < dimension; i++ {
		dot += float64(query[i]) * float64(node.projection[i])
	}
	if math.Abs(dot-node.threshold) < margin {
		leftIDs := searchTreeMultiProbeWithMargin(node.left, query, dimension, distance, margin)
		rightIDs := searchTreeMultiProbeWithMargin(node.right, query, dimension, distance, margin)
		return append(leftIDs, rightIDs...)
	} else if dot < node.threshold {
		return searchTreeMultiProbeWithMargin(node.left, query, dimension, distance, margin)
	}
	return searchTreeMultiProbeWithMargin(node.right, query, dimension, distance, margin)
}

// unionInts returns the union of two slices of ints.
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

// computeDistances computes the distance from the query to each vector corresponding to the provided ids.
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

// Search returns the k nearest neighbors for a query vector.
func (r *RPTIndex) Search(query []float32, k int) ([]core.Neighbor, error) {
	r.mu.RLock()
	if len(query) != r.dimension {
		r.mu.RUnlock()
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), r.dimension)
	}
	if len(r.points) == 0 {
		r.mu.RUnlock()
		return nil, errors.New("index is empty")
	}
	// Normalize the query if using cosine distance.
	queryCopy := make([]float32, len(query))
	copy(queryCopy, query)
	if r.DistanceName == "cosine" {
		core.NormalizeVector(queryCopy)
	}
	query = queryCopy

	if r.dirty {
		r.mu.RUnlock()
		r.mu.Lock()
		if r.dirty {
			r.buildTree()
		}
		r.mu.Unlock()
		r.mu.RLock()
	}
	candidateIDs := searchTreeMultiProbeWithMargin(r.tree, query, r.dimension, r.Distance, probeMargin)
	if len(candidateIDs) < k*2 {
		candidateIDsAlt := searchTreeMultiProbeWithMargin(r.tree, query, r.dimension, r.Distance, probeMargin*2)
		candidateIDs = unionInts(candidateIDs, candidateIDsAlt)
	}
	r.mu.RUnlock()

	neighbors := r.computeDistances(query, candidateIDs)
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
	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})
	if k > len(neighbors) {
		k = len(neighbors)
	}
	return neighbors[:k], nil
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
	// Normalize vector if using cosine distance.
	if r.DistanceName == "cosine" {
		core.NormalizeVector(vector)
	}
	r.points[id] = vector
	r.dirty = true
	return nil
}

// BulkAdd inserts multiple vectors into the index.
func (r *RPTIndex) BulkAdd(vectors map[int][]float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	// If using cosine distance, perform bulk normalization.
	if r.DistanceName == "cosine" {
		var vecs [][]float32
		for id, vector := range vectors {
			if len(vector) != r.dimension {
				return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), r.dimension, id)
			}
			vecs = append(vecs, vector)
		}
		core.NormalizeBatch(vecs)
	}
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
		delete(r.points, id)
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
	if r.DistanceName == "cosine" {
		core.NormalizeVector(vector)
	}
	r.points[id] = vector
	r.dirty = true
	return nil
}

// BulkUpdate updates multiple vectors in the index.
func (r *RPTIndex) BulkUpdate(updates map[int][]float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	// If using cosine distance, perform bulk normalization.
	if r.DistanceName == "cosine" {
		var vecs [][]float32
		for id, vector := range updates {
			if len(vector) != r.dimension {
				return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), r.dimension, id)
			}
			vecs = append(vecs, vector)
		}
		core.NormalizeBatch(vecs)
	}
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

// Stats returns metadata about the index.
func (r *RPTIndex) Stats() core.IndexStats {
	r.mu.RLock()
	defer r.mu.RUnlock()
	count := len(r.points)
	return core.IndexStats{
		Count:     count,
		Dimension: r.dimension,
		Distance:  r.DistanceName,
	}
}

// ----- Persistence -----
//
// We persist only the points and dimension; the tree is rebuilt on demand.
type rptSerialized struct {
	Dimension    int
	Points       map[int][]float32
	DistanceName string
}

func (r *RPTIndex) GobEncode() ([]byte, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ser := rptSerialized{
		Dimension:    r.dimension,
		Points:       r.points,
		DistanceName: r.DistanceName,
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
	r.DistanceName = ser.DistanceName
	r.dirty = true
	return nil
}

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

var _ core.Index = (*RPTIndex)(nil)

func init() {
	gob.Register(&RPTIndex{})
}

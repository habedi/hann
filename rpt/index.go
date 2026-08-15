package rpt

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/habedi/hann/core"
)

// Default values for the tuning parameters accepted by New.
const (
	defaultLeafCapacity         = 10
	defaultCandidateProjections = 3
	defaultParallelThreshold    = 100
	defaultProbeMargin          = 1.0
)

// rebuildFloor is the smallest number of overlay entries (pending adds plus
// stale tree references) that a mutation must reach before the tree is
// rebuilt. Below it, small indexes are served from the overlay alone.
const rebuildFloor = 64

// Option configures an Index created by New.
type Option func(*Index)

// WithLeafCapacity sets the maximum number of points stored in a tree leaf.
func WithLeafCapacity(leafCapacity int) Option {
	return func(r *Index) { r.leafCapacity = leafCapacity }
}

// WithCandidateProjections sets the number of random projections tried when
// splitting a tree node.
func WithCandidateProjections(candidateProjections int) Option {
	return func(r *Index) { r.candidateProjections = candidateProjections }
}

// WithParallelThreshold sets the point count above which subtrees are built in
// parallel.
func WithParallelThreshold(parallelThreshold int) Option {
	return func(r *Index) { r.parallelThreshold = parallelThreshold }
}

// WithProbeMargin sets the margin within which a search probes both children
// of a tree node. The margin is a fraction of the interquartile range of the
// projection values the node saw at build time.
func WithProbeMargin(probeMargin float64) Option {
	return func(r *Index) { r.probeMargin = probeMargin }
}

// WithBruteForceFallback sets whether a search that finds too few candidates
// falls back to a brute-force scan over all points.
func WithBruteForceFallback(allow bool) Option {
	return func(r *Index) { r.allowBruteForceFallback = allow }
}

// WithMetric sets the metric used to compare vectors.
func WithMetric(metric core.Metric) Option {
	return func(r *Index) { r.metric = metric }
}

// New creates an RPT (Random Projection Tree) index for vectors of the given
// dimension. Tuning parameters are set through options. The defaults are a
// leaf capacity of 10, 3 candidate projections, a parallel threshold of 100,
// a probe margin of 1.0, brute-force fallback enabled, and the Euclidean
// metric. It returns an error when the dimension, an option value, or the
// metric is invalid.
func New(dimension int, opts ...Option) (*Index, error) {
	r := &Index{
		dimension:               dimension,
		points:                  make(map[int][]float32),
		pendingAdds:             make(map[int]struct{}),
		leafCapacity:            defaultLeafCapacity,
		candidateProjections:    defaultCandidateProjections,
		parallelThreshold:       defaultParallelThreshold,
		probeMargin:             defaultProbeMargin,
		metric:                  core.Euclidean,
		allowBruteForceFallback: true,
	}
	for _, opt := range opts {
		opt(r)
	}
	if dimension <= 0 {
		return nil, fmt.Errorf("dimension (%d) must be positive", dimension)
	}
	// A non-positive leaf capacity makes tree building recurse forever, and a
	// non-positive number of candidate projections leaves a split with no
	// candidate.
	if r.leafCapacity <= 0 {
		return nil, fmt.Errorf("leaf capacity (%d) must be positive", r.leafCapacity)
	}
	if r.candidateProjections <= 0 {
		return nil, fmt.Errorf("candidate projections (%d) must be positive", r.candidateProjections)
	}
	if r.parallelThreshold <= 0 {
		return nil, fmt.Errorf("parallel threshold (%d) must be positive", r.parallelThreshold)
	}
	if r.probeMargin < 0 {
		return nil, fmt.Errorf("probe margin (%g) must be non-negative", r.probeMargin)
	}
	if r.metric.IsZero() {
		return nil, fmt.Errorf("metric must not be the zero value")
	}
	return r, nil
}

// treeNode represents a node in the random projection tree.
// It holds the projection, threshold, and pointers to left/right children.
// If isLeaf is true, the node holds a list of point ids.
type treeNode struct {
	isLeaf     bool      // true if this node is a leaf
	points     []int     // ids of points in the leaf
	projection []float32 // projection vector used for splitting at this node
	threshold  float64   // split threshold (median value)
	spread     float64   // interquartile range of the projection values at build time
	left       *treeNode // left child node
	right      *treeNode // right child node
}

// Index is the main structure for the random projection tree index.
// It holds all points, the tree root, and configuration parameters.
//
// The tree is rebuilt by writers, not by searches. Mutations record their
// effect in an overlay: pendingAdds holds ids the tree does not contain yet,
// and staleCount counts ids the tree references whose entry in the point map
// was removed or replaced. When the overlay grows past a threshold, the
// mutation that crossed it rebuilds the tree while still holding the write
// lock. This spreads the rebuild cost across mutations, and it keeps Search
// under a single read lock.
type Index struct {
	mu                      sync.RWMutex      // protects concurrent access
	fallbackSearches        atomic.Int64      // searches that scanned all points (see Search)
	dimension               int               // dimension of each vector
	points                  map[int][]float32 // mapping of point id to vector
	tree                    *treeNode         // root of the random projection tree, nil before the first build
	pendingAdds             map[int]struct{}  // ids added since the tree was last built
	staleCount              int               // ids removed or replaced that the tree may still reference
	metric                  core.Metric       // metric used to compare vectors
	leafCapacity            int               // maximum number of points in a leaf
	candidateProjections    int               // number of random projections to try when splitting
	parallelThreshold       int               // threshold to trigger parallel tree building
	probeMargin             float64           // margin for multi-probe search
	allowBruteForceFallback bool              // whether to allow falling back to a full brute-force scan
}

// buildTreeRecursive builds the tree recursively using random projections.
// It splits the given set of point ids based on a randomly chosen projection.
func buildTreeRecursive(ids []int, points map[int][]float32, dimension int,
	rnd *rand.Rand,
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
		spread    float64   // interquartile range of the projection values
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
		// Record the interquartile range of the projection values. The
		// probe margin is scaled by it at query time.
		spread := pairs[(3*len(pairs))/4].dot - pairs[len(pairs)/4].dot

		// Choose a random point x and compute the maximum distance to any other point.
		x := points[ids[rnd.Intn(len(ids))]]
		var maxDist float64
		for _, id := range ids {
			y := points[id]
			var dist float64
			for i := 0; i < dimension; i++ {
				d := float64(x[i] - y[i])
				dist += d * d
			}
			if dist > maxDist {
				maxDist = dist
			}
		}
		maxDist = math.Sqrt(maxDist)

		// Compute jitter
		jitter := (rnd.Float64()*2 - 1) * 6 * maxDist / math.Sqrt(float64(dimension))

		// Median threshold with jitter
		threshold := pairs[mid].dot + jitter

		// Split ids into left and right groups.
		var leftIDs, rightIDs []int
		for _, p := range pairs {
			if p.dot < threshold {
				leftIDs = append(leftIDs, p.id)
			} else {
				rightIDs = append(rightIDs, p.id)
			}
		}
		// Fallback: the jitter pushed the threshold outside the range of
		// the projection values, so one side is empty. Split at the median
		// value instead. The threshold stored in the node then still sends
		// a query to the child that holds its neighbors.
		if len(leftIDs) == 0 || len(rightIDs) == 0 {
			threshold = pairs[mid].dot
			leftIDs, rightIDs = nil, nil
			for _, p := range pairs {
				if p.dot < threshold {
					leftIDs = append(leftIDs, p.id)
				} else {
					rightIDs = append(rightIDs, p.id)
				}
			}
		}
		// Last resort: every projection value is equal, so no threshold on
		// this projection separates the points. Split evenly by position.
		// Routing is arbitrary among points with identical projections.
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
			spread:    spread,
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

		leftSeed := rnd.Int63()
		rightSeed := rnd.Int63()
		leftRnd := rand.New(rand.NewSource(leftSeed))
		rightRnd := rand.New(rand.NewSource(rightSeed))

		go func() {
			defer wg.Done()
			leftChild = buildTreeRecursive(bestCandidate.leftIDs, points, dimension,
				leftRnd, leafCapacity, candidateProjections, parallelThreshold)
		}()
		go func() {
			defer wg.Done()
			rightChild = buildTreeRecursive(bestCandidate.rightIDs, points, dimension,
				rightRnd, leafCapacity, candidateProjections, parallelThreshold)
		}()
		wg.Wait()
	} else {
		// Otherwise, build recursively in a single thread.
		leftChild = buildTreeRecursive(bestCandidate.leftIDs, points, dimension, rnd,
			leafCapacity, candidateProjections, parallelThreshold)
		rightChild = buildTreeRecursive(bestCandidate.rightIDs, points, dimension, rnd,
			leafCapacity, candidateProjections, parallelThreshold)
	}

	// Return an internal node with the best projection and split.
	return &treeNode{
		isLeaf:     false,
		projection: bestCandidate.proj,
		threshold:  bestCandidate.threshold,
		spread:     bestCandidate.spread,
		left:       leftChild,
		right:      rightChild,
	}
}

// buildTree constructs the random projection tree from all stored points and
// resets the overlay, since the new tree covers every stored point. The caller
// must hold the write lock.
func (r *Index) buildTree() {
	// Collect all point ids.
	ids := make([]int, 0, len(r.points))
	for id := range r.points {
		ids = append(ids, id)
	}
	// Use a new random source for building the tree.
	localRand := rand.New(rand.NewSource(core.GetSeed()))
	// Sort the ids to remove the map iteration order, then shuffle them with
	// the seeded generator. A run with HANN_SEED set then builds the same
	// tree.
	sort.Ints(ids)
	localRand.Shuffle(len(ids), func(i, j int) {
		ids[i], ids[j] = ids[j], ids[i]
	})
	r.tree = buildTreeRecursive(ids, r.points, r.dimension, localRand, r.leafCapacity,
		r.candidateProjections, r.parallelThreshold)
	r.pendingAdds = make(map[int]struct{})
	r.staleCount = 0
}

// maybeRebuild rebuilds the tree when the overlay has grown past the rebuild
// threshold. The threshold is the larger of rebuildFloor and a quarter of
// the stored points, counted as pending adds plus stale references. Every
// mutation calls it before releasing the write lock, which the caller must
// hold.
func (r *Index) maybeRebuild() {
	threshold := len(r.points) / 4
	if threshold < rebuildFloor {
		threshold = rebuildFloor
	}
	if len(r.pendingAdds)+r.staleCount > threshold {
		r.buildTree()
	}
}

// searchTreeMultiProbeWithMargin searches the tree for candidate point ids using multi-probing.
// It follows both branches if the projection value is close to the threshold (within margin).
func searchTreeMultiProbeWithMargin(node *treeNode, query []float32, dimension int,
	margin float64) []int {
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
	// If close to the threshold, probe both children. The margin is a
	// fraction of the interquartile range of the projection values seen at
	// build time. Its effect therefore does not depend on the scale of the
	// data.
	if math.Abs(dot-node.threshold) < margin*node.spread {
		leftIDs := searchTreeMultiProbeWithMargin(node.left, query, dimension, margin)
		rightIDs := searchTreeMultiProbeWithMargin(node.right, query, dimension, margin)
		// Merge into a fresh slice. Appending to leftIDs could write into the
		// backing array of a leaf, which is shared between concurrent searches.
		merged := make([]int, 0, len(leftIDs)+len(rightIDs))
		merged = append(merged, leftIDs...)
		merged = append(merged, rightIDs...)
		return merged
	} else if dot < node.threshold {
		return searchTreeMultiProbeWithMargin(node.left, query, dimension, margin)
	}
	return searchTreeMultiProbeWithMargin(node.right, query, dimension, margin)
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

// computeDistances calculates the rank distance from the query to each point
// id in the list. It does this in parallel across available CPUs. The
// Distance fields of the returned neighbors carry rank values. Rank values
// order candidates exactly like true distances. Search converts the final
// selection to true distances through the metric's FromRank before returning.
func (r *Index) computeDistances(query []float32, ids []int) ([]core.Neighbor, error) {
	// The tree can reference deleted ids until the next rebuild, so drop ids
	// that are no longer in the point map.
	present := make([]int, 0, len(ids))
	for _, id := range ids {
		if _, ok := r.points[id]; ok {
			present = append(present, id)
		}
	}
	ids = present
	neighbors := make([]core.Neighbor, len(ids))
	numWorkers := runtime.NumCPU()
	chunkSize := (len(ids) + numWorkers - 1) / numWorkers
	errsCh := make(chan error, numWorkers)

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
				d, err := r.metric.Rank(query, vec)
				if err != nil {
					errsCh <- err
					return
				}
				neighbors[j] = core.Neighbor{ID: id, Distance: d}
			}
		}(start, end)
	}
	wg.Wait()
	close(errsCh)

	for err := range errsCh {
		if err != nil {
			return nil, err // Return the first error encountered
		}
	}

	return neighbors, nil
}

// Search returns the k nearest neighbors to the query vector. It probes the
// tree for candidate ids and merges in the ids added since the tree was last
// built. It holds the read lock for its whole duration: rebuilding is the
// writers' job. When the tree was never built, the search is a scan of all
// stored points and counts as a fallback search in Stats.
func (r *Index) Search(query []float32, k int) ([]core.Neighbor, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}
	if len(query) != r.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d",
			len(query), r.dimension)
	}
	if len(r.points) == 0 {
		return nil, nil // Return empty slice for empty index
	}
	// Copy the query to avoid modifying the original.
	queryCopy := make([]float32, len(query))
	copy(queryCopy, query)
	query = queryCopy

	// A nil tree means no build has happened yet. The pending-add overlay
	// below then covers every stored point, and the search is a full scan.
	if r.tree == nil {
		r.fallbackSearches.Add(1)
	}
	// Get candidate ids using multi-probe search.
	candidateIDs := searchTreeMultiProbeWithMargin(r.tree, query, r.dimension, r.probeMargin)
	// If not enough candidates, try with a larger margin.
	if len(candidateIDs) < k*2 {
		candidateIDsAlt := searchTreeMultiProbeWithMargin(r.tree, query, r.dimension, r.probeMargin*2)
		candidateIDs = unionInts(candidateIDs, candidateIDsAlt)
	}
	// Merge in the ids the tree does not contain yet. The rebuild threshold
	// bounds their number, so this scan stays a fraction of the index.
	if len(r.pendingAdds) > 0 {
		pending := make([]int, 0, len(r.pendingAdds))
		for id := range r.pendingAdds {
			pending = append(pending, id)
		}
		candidateIDs = unionInts(candidateIDs, pending)
	}

	// Compute distances for candidate points. The read lock stays held so the
	// workers can read the point map while other goroutines mutate the index.
	neighbors, err := r.computeDistances(query, candidateIDs)
	if err != nil {
		return nil, err
	}
	// If still not enough, add extra points.
	if len(neighbors) < k {
		if !r.allowBruteForceFallback {
			// Return what we have, even if it's less than k
			sort.Slice(neighbors, func(i, j int) bool {
				return neighbors[i].Distance < neighbors[j].Distance
			})
			if k > len(neighbors) {
				k = len(neighbors)
			}
			return r.toTrueDistances(neighbors[:k]), nil
		}
		r.fallbackSearches.Add(1)
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
		extraNeighbors, err := r.computeDistances(query, missingIDs)
		if err != nil {
			return nil, err
		}
		neighbors = append(neighbors, extraNeighbors...)
	}
	// Sort by distance.
	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})
	if k > len(neighbors) {
		k = len(neighbors)
	}
	return r.toTrueDistances(neighbors[:k]), nil
}

// toTrueDistances converts the Distance field of each neighbor from a rank
// value to the true distance. It is applied exactly once, to the final k
// neighbors a search returns. Comparisons inside the index stay in rank
// space.
func (r *Index) toTrueDistances(neighbors []core.Neighbor) []core.Neighbor {
	for i := range neighbors {
		neighbors[i].Distance = r.metric.FromRank(neighbors[i].Distance)
	}
	return neighbors
}

// addLocked records a new id in the point map and the pending-add overlay.
// The caller must hold the write lock.
func (r *Index) addLocked(id int, vector []float32) {
	r.points[id] = vector
	r.pendingAdds[id] = struct{}{}
}

// deleteLocked removes an id from the point map and the overlay. When the
// tree never held the id, the id is dropped from the overlay too. When the
// tree holds the id, the id becomes a stale reference. The caller must hold
// the write lock.
func (r *Index) deleteLocked(id int) {
	delete(r.points, id)
	if _, pending := r.pendingAdds[id]; pending {
		delete(r.pendingAdds, id)
	} else {
		r.staleCount++
	}
}

// updateLocked replaces the vector of an id. When the tree holds the id, the
// entry stays reachable by id, but its placement in the tree goes stale.
// That counts toward the rebuild threshold. A pending id is rescanned on
// every search anyway, so it needs no bookkeeping. The caller must hold the
// write lock.
func (r *Index) updateLocked(id int, vector []float32) {
	r.points[id] = vector
	if _, pending := r.pendingAdds[id]; !pending {
		r.staleCount++
	}
}

// Add inserts a new point with the given id and vector into the index. The
// point is searchable immediately through the pending-add overlay, and the
// tree is rebuilt when the overlay grows past the rebuild threshold.
func (r *Index) Add(id int, vector []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(vector) != r.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(vector), r.dimension)
	}
	if _, exists := r.points[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}
	r.addLocked(id, vector)
	r.maybeRebuild()
	return nil
}

// BulkAdd inserts multiple points into the index and rebuilds the tree when
// the pending-add overlay grows past the rebuild threshold.
func (r *Index) BulkAdd(vectors map[int][]float32) error {
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
		r.addLocked(id, vector)
	}
	r.maybeRebuild()
	return nil
}

// Delete removes a point by its id and rebuilds the tree when the overlay
// grows past the rebuild threshold.
func (r *Index) Delete(id int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.points[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	r.deleteLocked(id)
	r.maybeRebuild()
	return nil
}

// BulkDelete removes multiple points from the index, ignoring ids that are
// not present, and rebuilds the tree when the overlay grows past the rebuild
// threshold.
func (r *Index) BulkDelete(ids []int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, id := range ids {
		if _, exists := r.points[id]; exists {
			r.deleteLocked(id)
		}
	}
	r.maybeRebuild()
	return nil
}

// Update changes the vector of an existing point and rebuilds the tree when
// the overlay grows past the rebuild threshold.
func (r *Index) Update(id int, vector []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(vector) != r.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(vector), r.dimension)
	}
	if _, exists := r.points[id]; !exists {
		return fmt.Errorf("id %d not found", id)
	}
	r.updateLocked(id, vector)
	r.maybeRebuild()
	return nil
}

// BulkUpdate updates multiple points in the index and rebuilds the tree when
// the overlay grows past the rebuild threshold.
func (r *Index) BulkUpdate(updates map[int][]float32) error {
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
		r.updateLocked(id, vector)
	}
	r.maybeRebuild()
	return nil
}

// Stats returns some basic statistics about the index.
func (r *Index) Stats() core.IndexStats {
	r.mu.RLock()
	defer r.mu.RUnlock()
	count := len(r.points)
	return core.IndexStats{
		Count:            count,
		Dimension:        r.dimension,
		Distance:         r.metric.Name(),
		FallbackSearches: r.fallbackSearches.Load(),
	}
}

// rptSerialized is used to serialize the index using gob.
type rptSerialized struct {
	Dimension               int
	Points                  map[int][]float32
	DistanceName            string
	LeafCapacity            int
	CandidateProjections    int
	ParallelThreshold       int
	ProbeMargin             float64
	AllowBruteForceFallback bool
	FormatVersion           int
}

// formatVersion is the on-disk format version written by GobEncode. Files
// written before the field existed decode it as zero and are accepted. Files
// written by a newer version of the format are rejected on load.
const formatVersion = 1

// GobEncode serializes the index to bytes using gob.
func (r *Index) GobEncode() ([]byte, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ser := rptSerialized{
		Dimension:               r.dimension,
		Points:                  r.points,
		DistanceName:            r.metric.Name(),
		LeafCapacity:            r.leafCapacity,
		CandidateProjections:    r.candidateProjections,
		ParallelThreshold:       r.parallelThreshold,
		ProbeMargin:             r.probeMargin,
		AllowBruteForceFallback: r.allowBruteForceFallback,
		FormatVersion:           formatVersion,
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(ser); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode deserializes the index from gob data.
func (r *Index) GobDecode(data []byte) error {
	var ser rptSerialized
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&ser); err != nil {
		return err
	}
	if ser.FormatVersion > formatVersion {
		return fmt.Errorf("index file has format version %d, but this build supports up to %d",
			ser.FormatVersion, formatVersion)
	}
	r.dimension = ser.Dimension
	r.points = ser.Points
	// Restore the metric from its name. On an unknown name, keep the metric
	// that is already configured. When there is none to keep, return an
	// error.
	if m, ok := core.MetricByName(ser.DistanceName); ok {
		r.metric = m
	} else if r.metric.IsZero() {
		return fmt.Errorf("unknown metric %q in serialized index", ser.DistanceName)
	}
	r.leafCapacity = ser.LeafCapacity
	r.candidateProjections = ser.CandidateProjections
	r.parallelThreshold = ser.ParallelThreshold
	r.probeMargin = ser.ProbeMargin
	r.allowBruteForceFallback = ser.AllowBruteForceFallback
	r.pendingAdds = make(map[int]struct{})
	r.staleCount = 0
	r.tree = nil
	// Build the tree now, so a loaded index searches without a first-search
	// rebuild. The build draws its randomness from core.GetSeed here instead
	// of at the first search. Under HANN_SEED that is the same draw.
	if len(r.points) > 0 {
		r.buildTree()
	}
	return nil
}

// Save writes the index to the given writer using gob encoding.
// GobEncode takes the read lock, so Save must not take it as well. A writer
// queued between the two acquisitions would deadlock the index.
func (r *Index) Save(w io.Writer) error {
	enc := gob.NewEncoder(w)
	return enc.Encode(r)
}

// Load reads the index from the given reader using gob encoding.
func (r *Index) Load(rdr io.Reader) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	dec := gob.NewDecoder(rdr)
	return dec.Decode(r)
}

// Check that Index implements the core.Index and core.BulkIndex interfaces.
var (
	_ core.Index     = (*Index)(nil)
	_ core.BulkIndex = (*Index)(nil)
)

// Register Index for gob encoding.
func init() {
	gob.Register(&Index{})
}

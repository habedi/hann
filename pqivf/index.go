package pqivf

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/habedi/hann/core"
)

// seededRand is a global random number generator for random operations,
// such as k-means seeding.
var seededRand = rand.New(rand.NewSource(core.GetSeed()))
var seededRandMu sync.Mutex

// pqEntry represents an entry in the index with its vector, PQ codes, and cluster assignment.
type pqEntry struct {
	ID      int       // unique identifier for the entry
	Vector  []float32 // original vector
	Codes   []int     // PQ codes for subquantizers (if trained)
	Cluster int       // coarse cluster assignment
}

// Index is the PQIVF index. It supports only the Euclidean metric. Coarse
// clustering and product quantization average vectors into k-means
// centroids, and that averaging assumes the Euclidean metric.
type Index struct {
	mu                      sync.RWMutex      // mutex for concurrent access
	fallbackSearches        atomic.Int64      // searches that fell back to a brute-force scan
	dimension               int               // dimension of the vectors
	coarseK                 int               // number of coarse clusters
	coarseCentroids         [][]float32       // centroids for coarse quantization
	clusterCounts           map[int]int       // count of vectors in each cluster
	invertedLists           map[int][]pqEntry // inverted index mapping clusters to entries
	numSubquantizers        int               // number of subquantizers (splits per vector)
	codebooks               [][][]float32     // codebooks for each subquantizer
	pqK                     int               // number of centroids per subquantizer (PQ codebook size)
	kMeansIters             int               // number of iterations for training the subquantizers
	idToCluster             map[int]int       // mapping from vector id to its cluster assignment
	metric                  core.Metric       // metric used for distance computation, fixed to Euclidean
	numCandidateClusters    int               // number of candidate clusters to consider during search
	allowBruteForceFallback bool              // whether to allow falling back to a full brute-force scan
	trained                 bool
	pendingVectors          map[int][]float32 // temporary holding area for vectors before training
}

// Option configures an index created by New.
type Option func(*Index)

// WithCoarseK sets the number of coarse clusters. The default is 16.
func WithCoarseK(coarseK int) Option {
	return func(pq *Index) { pq.coarseK = coarseK }
}

// WithNumSubquantizers sets the number of subquantizers. Each vector is
// split into that many parts. The dimension must be divisible by this value.
// The default is the largest of 8, 4, 2, and 1 that divides the dimension.
func WithNumSubquantizers(numSubquantizers int) Option {
	return func(pq *Index) { pq.numSubquantizers = numSubquantizers }
}

// WithPQK sets the number of centroids per subquantizer, which is the PQ
// codebook size. The default is 16.
func WithPQK(pqK int) Option {
	return func(pq *Index) { pq.pqK = pqK }
}

// WithKMeansIters sets the number of k-means iterations used during training.
// The default is 10.
func WithKMeansIters(iters int) Option {
	return func(pq *Index) { pq.kMeansIters = iters }
}

// WithCandidateClusters sets the number of candidate clusters probed during
// search. The default is 3.
func WithCandidateClusters(n int) Option {
	return func(pq *Index) { pq.numCandidateClusters = n }
}

// WithBruteForceFallback sets whether a search may fall back to a
// brute-force scan over all entries. The fallback runs when the probed
// clusters give fewer than k candidates. The fallback is on by default.
func WithBruteForceFallback(allow bool) Option {
	return func(pq *Index) { pq.allowBruteForceFallback = allow }
}

// New creates a new PQIVF index for vectors of the given dimension. The
// index always uses the Euclidean metric, because k-means centroid averaging
// assumes it. Defaults: 16 coarse clusters, a PQ codebook size of 16, 10
// k-means iterations, 3 candidate clusters probed per search, the brute-force
// fallback on, and the number of subquantizers set to the largest of 8, 4, 2,
// and 1 that divides the dimension. New returns an error in two cases. The
// first case is a non-positive value for the dimension, the number of coarse
// clusters, the PQ codebook size, or the number of k-means iterations. The
// second case is a dimension that is not divisible by the number of
// subquantizers.
func New(dimension int, opts ...Option) (*Index, error) {
	if dimension <= 0 {
		return nil, fmt.Errorf("dimension (%d) must be positive", dimension)
	}
	pq := &Index{
		dimension:               dimension,
		coarseK:                 16,
		coarseCentroids:         make([][]float32, 0),
		clusterCounts:           make(map[int]int),
		invertedLists:           make(map[int][]pqEntry),
		numSubquantizers:        defaultNumSubquantizers(dimension),
		codebooks:               nil,
		pqK:                     16,
		kMeansIters:             10,
		idToCluster:             make(map[int]int),
		metric:                  core.Euclidean,
		numCandidateClusters:    3,
		allowBruteForceFallback: true,
		trained:                 false,
		pendingVectors:          make(map[int][]float32),
	}
	for _, opt := range opts {
		opt(pq)
	}
	if pq.coarseK <= 0 {
		return nil, fmt.Errorf("coarseK (%d) must be positive", pq.coarseK)
	}
	if pq.pqK <= 0 {
		return nil, fmt.Errorf("pqK (%d) must be positive", pq.pqK)
	}
	if pq.kMeansIters <= 0 {
		return nil, fmt.Errorf("kMeansIters (%d) must be positive", pq.kMeansIters)
	}
	if pq.numSubquantizers <= 0 {
		return nil, fmt.Errorf("numSubquantizers (%d) must be positive", pq.numSubquantizers)
	}
	if dimension%pq.numSubquantizers != 0 {
		return nil, fmt.Errorf("dimension (%d) must be divisible by numSubquantizers (%d)", dimension, pq.numSubquantizers)
	}
	return pq, nil
}

// defaultNumSubquantizers returns the largest of 8, 4, 2, and 1 that divides
// the dimension.
func defaultNumSubquantizers(dimension int) int {
	for _, n := range []int{8, 4, 2} {
		if dimension%n == 0 {
			return n
		}
	}
	return 1
}

// nearestCentroid finds the closest coarse centroid to the vector. It
// returns the centroid's index and its rank distance. The distance is only
// used for ordering, so it stays in rank space.
func (pq *Index) nearestCentroid(vector []float32) (int, float64, error) {
	if len(pq.coarseCentroids) == 0 {
		return 0, 0, fmt.Errorf("no coarse centroids available")
	}
	best := -1
	bestDist := math.MaxFloat64
	for i, centroid := range pq.coarseCentroids {
		d, err := pq.metric.Rank(vector, centroid)
		if err != nil {
			return 0, 0, err
		}
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best, bestDist, nil
}

// nearestCentroids returns a sorted slice of clusters with their rank
// distances to the vector. The distances are only used for ordering, so
// they stay in rank space.
func (pq *Index) nearestCentroids(vector []float32) ([]struct {
	cluster int
	dist    float64
}, error) {
	res := make([]struct {
		cluster int
		dist    float64
	}, 0, len(pq.coarseCentroids))
	for i, centroid := range pq.coarseCentroids {
		d, err := pq.metric.Rank(vector, centroid)
		if err != nil {
			return nil, err
		}
		res = append(res, struct {
			cluster int
			dist    float64
		}{cluster: i, dist: d})
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i].dist < res[j].dist
	})
	return res, nil
}

// Add inserts a new vector with an id. On a trained index the vector goes
// into its nearest coarse cluster right away, and the index stays
// searchable. On an untrained index the vector waits in the temporary
// holding area until Train is called.
func (pq *Index) Add(id int, vector []float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	return pq.addLocked(id, vector)
}

// addLocked inserts a new vector with an id. When the index is trained, the
// vector is assigned to a cluster. Otherwise it goes into the pending list.
// The caller must hold the mutex.
func (pq *Index) addLocked(id int, vector []float32) error {
	if err := pq.validateVector(vector); err != nil {
		return err
	}
	if _, exists := pq.idToCluster[id]; exists {
		return fmt.Errorf("id %d already exists in a cluster", id)
	}
	if _, exists := pq.pendingVectors[id]; exists {
		return fmt.Errorf("id %d already exists in pending vectors", id)
	}

	if pq.trained {
		return pq.assignLocked(id, vector)
	}
	pq.pendingVectors[id] = vector
	return nil
}

// assignLocked places a vector into the trained index structure: the nearest
// coarse cluster, PQ codes when the codebooks allow encoding, and the id and
// count bookkeeping. Search accepts entries without codes and ranks them
// with the exact vector. A failed encoding therefore stores the entry
// without codes instead of failing the insert. The caller must hold the
// mutex, and the index must be trained.
func (pq *Index) assignLocked(id int, vector []float32) error {
	cluster, _, err := pq.nearestCentroid(vector)
	if err != nil {
		return err
	}
	entry := pqEntry{ID: id, Vector: vector, Cluster: cluster}
	if pq.codebooks != nil {
		if codes, err := pq.encodeVector(vector, cluster); err == nil {
			entry.Codes = codes
		}
	}
	pq.invertedLists[cluster] = append(pq.invertedLists[cluster], entry)
	pq.idToCluster[id] = cluster
	pq.clusterCounts[cluster]++
	return nil
}

// BulkAdd inserts multiple vectors. It follows the same rule as Add. On a
// trained index each vector goes into its nearest coarse cluster right away.
// On an untrained index the vectors wait in the temporary holding area until
// Train is called.
func (pq *Index) BulkAdd(vectors map[int][]float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	var keys []int
	for id := range vectors {
		keys = append(keys, id)
	}
	sort.Ints(keys)

	for _, id := range keys {
		vector := vectors[id]
		if len(vector) != pq.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), pq.dimension, id)
		}
		if _, exists := pq.idToCluster[id]; exists {
			return fmt.Errorf("id %d already exists in a cluster", id)
		}
		if _, exists := pq.pendingVectors[id]; exists {
			return fmt.Errorf("id %d already exists in pending vectors", id)
		}

		if pq.trained {
			if err := pq.assignLocked(id, vector); err != nil {
				return err
			}
		} else {
			pq.pendingVectors[id] = vector
		}
	}
	return nil
}

// Delete removes an entry by its id, from either pending vectors or clustered data.
func (pq *Index) Delete(id int) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	return pq.deleteLocked(id)
}

// deleteLocked removes an entry by its id, from either pending vectors or
// clustered data. A delete does not untrain the index and does not move the
// coarse centroids. The stored PQ codes are residuals against the centroid
// they were encoded with. Recomputing a centroid from the surviving entries
// would silently degrade every code in the cluster. So the centroids stay
// fixed between Train calls, and Train is the refresh. The caller must hold
// the mutex.
func (pq *Index) deleteLocked(id int) error {
	// If the vector is in the pending list, remove it from there.
	if _, exists := pq.pendingVectors[id]; exists {
		delete(pq.pendingVectors, id)
		return nil
	}

	// If not in pending, it must be in a cluster.
	cluster, exists := pq.idToCluster[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	entries, ok := pq.invertedLists[cluster]
	if !ok {
		return fmt.Errorf("inconsistent state: cluster %d not found for id %d", cluster, id)
	}
	found := false
	var newEntries []pqEntry
	for _, entry := range entries {
		if entry.ID == id {
			found = true
			pq.clusterCounts[cluster]--
		} else {
			newEntries = append(newEntries, entry)
		}
	}
	if !found {
		return fmt.Errorf("id %d not found in cluster %d", id, cluster)
	}
	pq.invertedLists[cluster] = newEntries
	delete(pq.idToCluster, id)
	return nil
}

// BulkDelete removes multiple entries from the index. Like Delete, it keeps
// the index trained and leaves the coarse centroids fixed. See deleteLocked
// for the reasoning.
func (pq *Index) BulkDelete(ids []int) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	// Sort a copy, so the caller's slice keeps its order.
	sorted := append([]int(nil), ids...)
	sort.Ints(sorted)
	for _, id := range sorted {
		// If in pending, just delete.
		if _, exists := pq.pendingVectors[id]; exists {
			delete(pq.pendingVectors, id)
			continue
		}
		// Otherwise, find in clusters.
		cluster, exists := pq.idToCluster[id]
		if !exists {
			continue
		}
		entries, ok := pq.invertedLists[cluster]
		if !ok {
			continue
		}
		var newEntries []pqEntry
		for _, entry := range entries {
			if entry.ID == id {
				pq.clusterCounts[cluster]--
			} else {
				newEntries = append(newEntries, entry)
			}
		}
		pq.invertedLists[cluster] = newEntries
		delete(pq.idToCluster, id)
	}
	return nil
}

// validateVector checks that a vector is compatible with the index
// dimension. The caller must hold the mutex.
func (pq *Index) validateVector(vector []float32) error {
	if pq.dimension == 0 {
		return fmt.Errorf("cannot add to a zero-dimension index")
	}
	if len(vector) != pq.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), pq.dimension)
	}
	return nil
}

// Update removes an entry and re-adds it with a new vector. Both steps run
// in one critical section. No other operation can see the entry in its
// deleted state in between. The new vector is validated before the delete,
// so a failed update leaves the index unchanged.
func (pq *Index) Update(id int, vector []float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if err := pq.validateVector(vector); err != nil {
		return err
	}
	if err := pq.deleteLocked(id); err != nil {
		return err
	}
	return pq.addLocked(id, vector)
}

// BulkUpdate updates multiple entries with new vectors. The whole batch runs
// in one critical section, so no other operation can run in the middle of
// it. All vectors are validated before any entry is touched, so a dimension
// mismatch leaves the index unchanged.
func (pq *Index) BulkUpdate(updates map[int][]float32) error {
	var keys []int
	for id := range updates {
		keys = append(keys, id)
	}
	sort.Ints(keys)

	pq.mu.Lock()
	defer pq.mu.Unlock()

	for _, id := range keys {
		if err := pq.validateVector(updates[id]); err != nil {
			return fmt.Errorf("invalid vector for id %d: %w", id, err)
		}
	}

	for _, id := range keys {
		if err := pq.deleteLocked(id); err != nil {
			return err
		}
		if err := pq.addLocked(id, updates[id]); err != nil {
			return err
		}
	}
	return nil
}

// Train builds the index structure, including coarse centroids and PQ
// codebooks. It is both the initial training step and an optional refresh.
// It re-clusters every vector, clustered and pending. Running it again
// restores clustering quality after many mutations. Once an index has been
// trained, later adds, deletes, and updates keep it searchable.
func (pq *Index) Train() error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	// Consolidate all vectors from pending and clustered lists.
	allVectorsByID := make(map[int][]float32)
	for _, entries := range pq.invertedLists {
		for _, entry := range entries {
			allVectorsByID[entry.ID] = entry.Vector
		}
	}
	for id, vector := range pq.pendingVectors {
		allVectorsByID[id] = vector
	}

	if len(allVectorsByID) < pq.coarseK {
		return fmt.Errorf("not enough vectors (%d) to train coarse quantizer with %d clusters", len(allVectorsByID), pq.coarseK)
	}

	var allVectors [][]float32
	for _, v := range allVectorsByID {
		allVectors = append(allVectors, v)
	}

	// Train coarse centroids using k-means on all available vectors.
	coarseCentroids, err := runKMeans(allVectors, pq.coarseK, pq.kMeansIters)
	if err != nil {
		return fmt.Errorf("failed to train coarse centroids: %w", err)
	}
	pq.coarseCentroids = coarseCentroids

	// Re-assign all vectors to the new coarse centroids.
	pq.invertedLists = make(map[int][]pqEntry)
	pq.idToCluster = make(map[int]int)
	pq.clusterCounts = make(map[int]int)
	for id, vector := range allVectorsByID {
		cluster, _, err := pq.nearestCentroid(vector)
		if err != nil {
			return err
		}
		pq.idToCluster[id] = cluster
		pq.clusterCounts[cluster]++
		entry := pqEntry{ID: id, Vector: vector, Cluster: cluster}
		pq.invertedLists[cluster] = append(pq.invertedLists[cluster], entry)
	}

	// Clear the pending vectors list as they are now clustered.
	pq.pendingVectors = make(map[int][]float32)

	// If there is no data, training is already complete.
	if len(pq.invertedLists) == 0 {
		pq.trained = true
		return nil
	}

	// Prepare data for subquantizer training by computing residuals.
	dataPerSub := make([][][]float32, pq.numSubquantizers)
	for i := 0; i < pq.numSubquantizers; i++ {
		dataPerSub[i] = make([][]float32, 0)
	}
	for cluster, entries := range pq.invertedLists {
		centroid := pq.coarseCentroids[cluster]
		for _, entry := range entries {
			residual, err := vectorSub(entry.Vector, centroid)
			if err != nil {
				return err
			}
			subVecs := splitVector(residual, pq.numSubquantizers)
			for i, sub := range subVecs {
				dataPerSub[i] = append(dataPerSub[i], sub)
			}
		}
	}

	// Train a codebook for each subquantizer.
	codebooks := make([][][]float32, pq.numSubquantizers)
	for i := 0; i < pq.numSubquantizers; i++ {
		cb, err := runKMeans(dataPerSub[i], pq.pqK, pq.kMeansIters)
		if err != nil {
			return err
		}
		codebooks[i] = cb
	}
	pq.codebooks = codebooks

	// Re-encode all entries with the new codebooks.
	for cluster, entries := range pq.invertedLists {
		for j, entry := range entries {
			codes, err := pq.encodeVector(entry.Vector, cluster)
			if err != nil {
				return err
			}
			entry.Codes = codes
			pq.invertedLists[cluster][j] = entry
		}
	}

	pq.trained = true
	return nil
}

// encodeVector computes the PQ codes for a vector given its coarse cluster.
func (pq *Index) encodeVector(vector []float32, cluster int) ([]int, error) {
	if pq.codebooks == nil {
		return nil, fmt.Errorf("codebooks not trained")
	}
	residual, err := vectorSub(vector, pq.coarseCentroids[cluster])
	if err != nil {
		return nil, err
	}
	subVecs := splitVector(residual, pq.numSubquantizers)
	codes := make([]int, pq.numSubquantizers)
	for i, sub := range subVecs {
		best := -1
		bestDist := math.MaxFloat64
		for j, cent := range pq.codebooks[i] {
			d, err := core.Euclidean.Rank(sub, cent)
			if err != nil {
				return nil, err
			}
			if d < bestDist {
				bestDist = d
				best = j
			}
		}
		if best < 0 {
			return nil, fmt.Errorf("failed to encode sub-vector")
		}
		codes[i] = best
	}
	return codes, nil
}

// decodePQCode reconstructs an approximate residual from the PQ codes.
func (pq *Index) decodePQCode(codes []int) ([]float32, error) {
	if pq.codebooks == nil {
		return nil, fmt.Errorf("codebooks not trained")
	}
	var approx []float32
	for i, code := range codes {
		if i >= len(pq.codebooks) || code >= len(pq.codebooks[i]) {
			return nil, fmt.Errorf("invalid PQ code")
		}
		approx = append(approx, pq.codebooks[i][code]...)
	}
	return approx, nil
}

// vectorSub computes the element-wise subtraction of two vectors.
func vectorSub(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("vector lengths do not match: %d vs %d", len(a), len(b))
	}
	res := make([]float32, len(a))
	for i := range a {
		res[i] = a[i] - b[i]
	}
	return res, nil
}

// vectorAdd computes the element-wise addition of two vectors.
func vectorAdd(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("vector lengths do not match: %d vs %d", len(a), len(b))
	}
	res := make([]float32, len(a))
	for i := range a {
		res[i] = a[i] + b[i]
	}
	return res, nil
}

// splitVector splits a vector into numParts equal parts.
func splitVector(vec []float32, numParts int) [][]float32 {
	total := len(vec)
	subDim := total / numParts
	parts := make([][]float32, numParts)
	start := 0
	for i := 0; i < numParts; i++ {
		end := start + subDim
		parts[i] = vec[start:end]
		start = end
	}
	return parts
}

// runKMeans runs a basic k-means clustering on the provided data.
func runKMeans(data [][]float32, k int, iterations int) ([][]float32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("no data for k-means training")
	}
	if len(data) < k {
		k = len(data)
	}
	centroids := make([][]float32, k)
	seededRandMu.Lock()
	perm := seededRand.Perm(len(data))
	seededRandMu.Unlock()
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, len(data[0]))
		copy(centroids[i], data[perm[i]])
	}
	for iter := 0; iter < iterations; iter++ {
		clusters := make([][][]float32, k)
		for i := range clusters {
			clusters[i] = make([][]float32, 0)
		}
		for _, point := range data {
			best := -1
			bestDist := math.MaxFloat64
			for i, cent := range centroids {
				d, err := core.Euclidean.Rank(point, cent)
				if err != nil {
					return nil, err
				}
				if d < bestDist {
					bestDist = d
					best = i
				}
			}
			clusters[best] = append(clusters[best], point)
		}
		for i, clusterData := range clusters {
			if len(clusterData) == 0 {
				// If a cluster is empty, reinitialize its centroid randomly.
				seededRandMu.Lock()
				index := seededRand.Intn(len(data))
				seededRandMu.Unlock()
				newCentroid := make([]float32, len(data[0]))
				copy(newCentroid, data[index])
				centroids[i] = newCentroid
			} else {
				newCentroid := make([]float32, len(data[0]))
				for _, point := range clusterData {
					for j, v := range point {
						newCentroid[j] += v
					}
				}
				for j := range newCentroid {
					newCentroid[j] /= float32(len(clusterData))
				}
				centroids[i] = newCentroid
			}
		}
	}
	return centroids, nil
}

// Search finds the k nearest neighbors for the given query vector.
func (pq *Index) Search(query []float32, k int) ([]core.Neighbor, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if k <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}

	if !pq.trained {
		return nil, fmt.Errorf("PQIVF index is not trained. Call Train() after adding data before searching")
	}

	if len(query) != pq.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), pq.dimension)
	}
	// Copy query to avoid modifying original vector.
	queryCopy := make([]float32, len(query))
	copy(queryCopy, query)
	query = queryCopy

	if len(pq.invertedLists) == 0 {
		return nil, fmt.Errorf("index is empty")
	}

	// Get nearest coarse centroids as candidate clusters.
	centCandidates, err := pq.nearestCentroids(query)
	if err != nil {
		return nil, fmt.Errorf("failed to find nearest centroids: %w", err)
	}
	numCandidates := pq.numCandidateClusters
	if numCandidates > len(centCandidates) {
		numCandidates = len(centCandidates)
	}
	var entries []pqEntry
	// Collect entries from the top candidate clusters.
	for i := 0; i < numCandidates; i++ {
		cluster := centCandidates[i].cluster
		entries = append(entries, pq.invertedLists[cluster]...)
	}

	// If the number of candidates is less than k, and fallback is allowed,
	// perform a brute-force scan over all entries.
	if len(entries) < k && pq.allowBruteForceFallback {
		pq.fallbackSearches.Add(1)
		var allEntries []pqEntry
		for _, list := range pq.invertedLists {
			allEntries = append(allEntries, list...)
		}
		entries = allEntries
	}

	var results []core.Neighbor
	// Compute rank distances for each candidate entry. Rank distances order
	// candidates exactly like true distances, so the sort below works on
	// them as is. Only the final k results are converted to true distances.
	for _, entry := range entries {
		var d float64
		var distErr error

		if pq.codebooks != nil && len(entry.Codes) == pq.numSubquantizers {
			approxResidual, err := pq.decodePQCode(entry.Codes)
			if err != nil {
				// Fallback to exact distance if decoding fails
				d, distErr = pq.metric.Rank(query, entry.Vector)
			} else {
				approxVec, err := vectorAdd(pq.coarseCentroids[entry.Cluster], approxResidual)
				if err != nil {
					// Fallback to exact distance if vector addition fails
					d, distErr = pq.metric.Rank(query, entry.Vector)
				} else {
					// Main path: use approximate distance
					d, distErr = pq.metric.Rank(query, approxVec)
				}
			}
		} else {
			// Path for entries without PQ codes, such as an entry added
			// after training whose encoding failed.
			d, distErr = pq.metric.Rank(query, entry.Vector)
		}

		// A candidate whose distance cannot be computed is skipped.
		if distErr != nil {
			continue
		}

		results = append(results, core.Neighbor{ID: entry.ID, Distance: d})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	if k > len(results) {
		k = len(results)
	}
	results = results[:k]
	// Convert the surviving rank distances to true distances exactly once.
	for i := range results {
		results[i].Distance = pq.metric.FromRank(results[i].Distance)
	}
	return results, nil
}

// Stats returns statistics about the index, such as the total number of
// entries.
func (pq *Index) Stats() core.IndexStats {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	count := len(pq.pendingVectors)
	for _, entries := range pq.invertedLists {
		count += len(entries)
	}
	return core.IndexStats{
		Count:            count,
		Dimension:        pq.dimension,
		Distance:         pq.metric.Name(),
		FallbackSearches: pq.fallbackSearches.Load(),
	}
}

// serializedPQIVF is a serializable representation of the PQIVF index.
type serializedPQIVF struct {
	Dimension               int
	CoarseK                 int
	CoarseCentroids         [][]float32
	ClusterCounts           map[int]int
	InvertedLists           map[int][]pqEntry
	NumSubquantizers        int
	Codebooks               [][][]float32
	PqK                     int
	KMeansIters             int
	AllowBruteForceFallback bool
	Trained                 bool
	PendingVectors          map[int][]float32
	FormatVersion           int
	NumCandidateClusters    int
}

// formatVersion is the on-disk format version written by GobEncode. Files
// written before the field existed decode it as zero, and they are accepted.
// Files written by a newer version of the format are rejected on load.
const formatVersion = 1

// GobEncode serializes the index into bytes using gob.
func (pq *Index) GobEncode() ([]byte, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	ser := serializedPQIVF{
		Dimension:               pq.dimension,
		CoarseK:                 pq.coarseK,
		CoarseCentroids:         pq.coarseCentroids,
		ClusterCounts:           pq.clusterCounts,
		InvertedLists:           pq.invertedLists,
		NumSubquantizers:        pq.numSubquantizers,
		Codebooks:               pq.codebooks,
		PqK:                     pq.pqK,
		KMeansIters:             pq.kMeansIters,
		AllowBruteForceFallback: pq.allowBruteForceFallback,
		Trained:                 pq.trained,
		PendingVectors:          pq.pendingVectors,
		FormatVersion:           formatVersion,
		NumCandidateClusters:    pq.numCandidateClusters,
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(ser); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode deserializes the index from bytes using gob.
func (pq *Index) GobDecode(data []byte) error {
	var ser serializedPQIVF
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&ser); err != nil {
		return err
	}
	if ser.FormatVersion > formatVersion {
		return fmt.Errorf("index file has format version %d, but this build supports up to %d",
			ser.FormatVersion, formatVersion)
	}
	// A corrupt or crafted file can carry parameters New would reject, and
	// some of them cause a panic in a later operation, such as a division
	// by a zero subquantizer count. Real files always carry valid values,
	// so an invalid one is an error, not a fallback.
	if ser.Dimension <= 0 {
		return fmt.Errorf("serialized index has invalid dimension %d", ser.Dimension)
	}
	if ser.NumSubquantizers <= 0 || ser.Dimension%ser.NumSubquantizers != 0 {
		return fmt.Errorf("serialized index has invalid subquantizer count %d for dimension %d",
			ser.NumSubquantizers, ser.Dimension)
	}
	if ser.CoarseK <= 0 || ser.PqK <= 0 || ser.KMeansIters <= 0 {
		return fmt.Errorf("serialized index has invalid training parameters: coarseK %d, pqK %d, kMeansIters %d",
			ser.CoarseK, ser.PqK, ser.KMeansIters)
	}
	pq.dimension = ser.Dimension
	pq.coarseK = ser.CoarseK
	pq.coarseCentroids = ser.CoarseCentroids
	pq.clusterCounts = ser.ClusterCounts
	if pq.clusterCounts == nil {
		pq.clusterCounts = make(map[int]int)
	}
	pq.invertedLists = ser.InvertedLists
	if pq.invertedLists == nil {
		pq.invertedLists = make(map[int][]pqEntry)
	}
	pq.numSubquantizers = ser.NumSubquantizers
	pq.codebooks = ser.Codebooks
	pq.pqK = ser.PqK
	pq.kMeansIters = ser.KMeansIters
	pq.allowBruteForceFallback = ser.AllowBruteForceFallback
	pq.trained = ser.Trained
	pq.pendingVectors = ser.PendingVectors
	if pq.pendingVectors == nil {
		pq.pendingVectors = make(map[int][]float32)
	}
	// Files written before the field existed decode it as zero. Keep the
	// configured value in that case, and fall back to the default of 3 when
	// there is none to keep.
	if ser.NumCandidateClusters > 0 {
		pq.numCandidateClusters = ser.NumCandidateClusters
	} else if pq.numCandidateClusters <= 0 {
		pq.numCandidateClusters = 3
	}
	pq.idToCluster = make(map[int]int)
	// Rebuild idToCluster mapping from the inverted lists.
	for cluster, entries := range pq.invertedLists {
		for _, entry := range entries {
			pq.idToCluster[entry.ID] = cluster
		}
	}
	pq.metric = core.Euclidean
	return nil
}

// Save writes the index to the given writer using gob encoding.
// Encoding goes through GobEncode, which takes the read lock, so Save
// must not take the lock itself. Taking the read lock twice can deadlock
// when a writer queues between the two acquisitions.
func (pq *Index) Save(w io.Writer) error {
	enc := gob.NewEncoder(w)
	return enc.Encode(pq)
}

// Load reads the index from the given reader using gob decoding.
func (pq *Index) Load(r io.Reader) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	dec := gob.NewDecoder(r)
	return dec.Decode(pq)
}

// Check interface compliance.
var _ core.Index = (*Index)(nil)
var _ core.BulkIndex = (*Index)(nil)
var _ core.Trainer = (*Index)(nil)

// init registers types for gob encoding.
func init() {
	gob.Register(&Index{})
	gob.Register(pqEntry{})
}

package pqivf

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/habedi/hann/core"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
)

// seededRand is a global random number generator for random operations (e.g. during k-means).
var seededRand = rand.New(rand.NewSource(core.GetSeed()))
var seededRandMu sync.Mutex

// pqEntry represents an entry in the index with its vector, PQ codes, and cluster assignment.
type pqEntry struct {
	ID      int       // unique identifier for the entry
	Vector  []float32 // original vector
	Codes   []int     // PQ codes for subquantizers (if trained)
	Cluster int       // coarse cluster assignment
}

// PQIVFIndex is the main structure for the PQIVF index.
type PQIVFIndex struct {
	mu                   sync.RWMutex      // mutex for concurrent access
	dimension            int               // dimension of the vectors
	coarseK              int               // number of coarse clusters
	coarseCentroids      [][]float32       // centroids for coarse quantization
	clusterCounts        map[int]int       // count of vectors in each cluster
	invertedLists        map[int][]pqEntry // inverted index mapping clusters to entries
	numSubquantizers     int               // number of subquantizers (splits per vector)
	codebooks            [][][]float32     // codebooks for each subquantizer
	pqK                  int               // number of centroids per subquantizer (PQ codebook size)
	kMeansIters          int               // number of iterations for training the subquantizers
	idToCluster          map[int]int       // mapping from vector id to its cluster assignment
	Distance             core.DistanceFunc // function to compute distance between vectors
	numCandidateClusters int               // number of candidate clusters to consider during search
}

// recalcCentroid recalculates the centroid for a given cluster based on its current entries.
func (pq *PQIVFIndex) recalcCentroid(cluster int) {
	entries := pq.invertedLists[cluster]
	if len(entries) == 0 {
		return
	}
	newCentroid := make([]float32, pq.dimension)
	for _, entry := range entries {
		for i, v := range entry.Vector {
			newCentroid[i] += v
		}
	}
	for i := range newCentroid {
		newCentroid[i] /= float32(len(entries))
	}
	pq.coarseCentroids[cluster] = newCentroid
}

// NewPQIVFIndex creates a new PQIVF index. It panics if the dimension is not divisible by numSubquantizers.
func NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters int) *PQIVFIndex {
	if dimension%numSubquantizers != 0 {
		panic(fmt.Sprintf("dimension (%d) must be divisible by numSubquantizers (%d)", dimension, numSubquantizers))
	}
	return &PQIVFIndex{
		dimension:            dimension,
		coarseK:              coarseK,
		coarseCentroids:      make([][]float32, 0),
		clusterCounts:        make(map[int]int),
		invertedLists:        make(map[int][]pqEntry),
		numSubquantizers:     numSubquantizers,
		codebooks:            nil,
		pqK:                  pqK,
		kMeansIters:          kMeansIters,
		idToCluster:          make(map[int]int),
		Distance:             core.Euclidean,
		numCandidateClusters: 3,
	}
}

// nearestCentroid finds the closest coarse centroid to the vector and returns its index and distance.
func (pq *PQIVFIndex) nearestCentroid(vector []float32) (int, float64) {
	best := -1
	bestDist := math.MaxFloat64
	for i, centroid := range pq.coarseCentroids {
		d := pq.Distance(vector, centroid)
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best, bestDist
}

// nearestCentroids returns a sorted slice of clusters with their distances to the vector.
func (pq *PQIVFIndex) nearestCentroids(vector []float32) []struct {
	cluster int
	dist    float64
} {
	res := make([]struct {
		cluster int
		dist    float64
	}, 0, len(pq.coarseCentroids))
	for i, centroid := range pq.coarseCentroids {
		d := pq.Distance(vector, centroid)
		res = append(res, struct {
			cluster int
			dist    float64
		}{cluster: i, dist: d})
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i].dist < res[j].dist
	})
	return res
}

// Add inserts a new vector with an id into the index.
func (pq *PQIVFIndex) Add(id int, vector []float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(vector) != pq.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), pq.dimension)
	}
	if _, exists := pq.idToCluster[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}

	var cluster int
	// If there aren't enough centroids yet, create a new one.
	if len(pq.coarseCentroids) < pq.coarseK {
		cluster = len(pq.coarseCentroids)
		centroid := make([]float32, pq.dimension)
		copy(centroid, vector)
		pq.coarseCentroids = append(pq.coarseCentroids, centroid)
		pq.clusterCounts[cluster] = 1
	} else {
		// Otherwise, assign to the nearest centroid.
		cluster, _ = pq.nearestCentroid(vector)
		pq.clusterCounts[cluster]++
	}

	pq.idToCluster[id] = cluster
	entry := pqEntry{ID: id, Vector: vector, Cluster: cluster}
	// If codebooks are available, encode the vector.
	if pq.codebooks != nil {
		codes, err := pq.encodeVector(vector, cluster)
		if err != nil {
			return err
		}
		entry.Codes = codes
	}
	pq.invertedLists[cluster] = append(pq.invertedLists[cluster], entry)
	pq.recalcCentroid(cluster)
	return nil
}

// BulkAdd inserts multiple vectors into the index.
func (pq *PQIVFIndex) BulkAdd(vectors map[int][]float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	var keys []int
	for id := range vectors {
		keys = append(keys, id)
	}
	sort.Ints(keys)
	updatedClusters := make(map[int]bool)
	for _, id := range keys {
		vector := vectors[id]
		if len(vector) != pq.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), pq.dimension, id)
		}
		if _, exists := pq.idToCluster[id]; exists {
			return fmt.Errorf("id %d already exists", id)
		}

		var cluster int
		// Create new centroid if needed.
		if len(pq.coarseCentroids) < pq.coarseK {
			cluster = len(pq.coarseCentroids)
			centroid := make([]float32, pq.dimension)
			copy(centroid, vector)
			pq.coarseCentroids = append(pq.coarseCentroids, centroid)
			pq.clusterCounts[cluster] = 1
		} else {
			cluster, _ = pq.nearestCentroid(vector)
			pq.clusterCounts[cluster]++
		}
		pq.idToCluster[id] = cluster
		var codes []int
		if pq.codebooks != nil {
			var err error
			codes, err = pq.encodeVector(vector, cluster)
			if err != nil {
				return err
			}
		}
		entry := pqEntry{ID: id, Vector: vector, Codes: codes, Cluster: cluster}
		pq.invertedLists[cluster] = append(pq.invertedLists[cluster], entry)
		updatedClusters[cluster] = true
	}
	// Recalculate centroids for clusters that got updated.
	for cluster := range updatedClusters {
		pq.recalcCentroid(cluster)
	}
	return nil
}

// Delete removes an entry by its id.
func (pq *PQIVFIndex) Delete(id int) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	cluster, exists := pq.idToCluster[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	entries, ok := pq.invertedLists[cluster]
	if !ok {
		return fmt.Errorf("inconsistent state: cluster %d not found", cluster)
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
	if len(newEntries) > 0 {
		pq.recalcCentroid(cluster)
	}
	return nil
}

// BulkDelete removes multiple entries from the index.
func (pq *PQIVFIndex) BulkDelete(ids []int) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	sort.Ints(ids)
	updatedClusters := make(map[int]bool)
	for _, id := range ids {
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
		if len(newEntries) > 0 {
			updatedClusters[cluster] = true
		}
	}
	// Recalculate centroids for updated clusters.
	for cluster := range updatedClusters {
		pq.recalcCentroid(cluster)
	}
	return nil
}

// Update removes and then re-adds an entry with an updated vector.
func (pq *PQIVFIndex) Update(id int, vector []float32) error {
	if err := pq.Delete(id); err != nil {
		return err
	}
	return pq.Add(id, vector)
}

// BulkUpdate updates multiple entries with new vectors.
func (pq *PQIVFIndex) BulkUpdate(updates map[int][]float32) error {
	var keys []int
	for id := range updates {
		keys = append(keys, id)
	}
	sort.Ints(keys)
	for _, id := range keys {
		vector := updates[id]
		if err := pq.Update(id, vector); err != nil {
			return err
		}
	}
	return nil
}

// Train runs k-means on residuals to train subquantizers (codebooks).
func (pq *PQIVFIndex) Train() error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(pq.invertedLists) == 0 {
		return fmt.Errorf("no data to train on")
	}

	// Prepare data for each subquantizer.
	dataPerSub := make([][][]float32, pq.numSubquantizers)
	for i := 0; i < pq.numSubquantizers; i++ {
		dataPerSub[i] = make([][]float32, 0)
	}

	// For each entry, compute residuals and split into sub-vectors.
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
		cb, err := trainSubquantizer(dataPerSub[i], pq.pqK, pq.kMeansIters)
		if err != nil {
			return err
		}
		codebooks[i] = cb
	}
	pq.codebooks = codebooks

	// Re-encode all entries using the new codebooks.
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

	return nil
}

// encodeVector computes the PQ codes for a vector given its coarse cluster.
func (pq *PQIVFIndex) encodeVector(vector []float32, cluster int) ([]int, error) {
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
			d := core.Euclidean(sub, cent)
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
func (pq *PQIVFIndex) decodePQCode(codes []int) ([]float32, error) {
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

// trainSubquantizer trains a codebook for a subquantizer using k-means.
func trainSubquantizer(data [][]float32, k int, iterations int) ([][]float32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("no data for subquantizer training")
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
				d := core.Euclidean(point, cent)
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
func (pq *PQIVFIndex) Search(query []float32, k int) ([]core.Neighbor, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

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
	centCandidates := pq.nearestCentroids(query)
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
	// If not enough entries, add more from further clusters.
	if len(entries) < k {
		for i := numCandidates; i < len(centCandidates) && len(entries) < k; i++ {
			cluster := centCandidates[i].cluster
			entries = append(entries, pq.invertedLists[cluster]...)
		}
	}
	// If still not enough, take all available entries.
	if len(entries) < k {
		var allEntries []pqEntry
		for _, list := range pq.invertedLists {
			allEntries = append(allEntries, list...)
		}
		entries = allEntries
	}

	var results []core.Neighbor
	// Compute distances for each candidate entry.
	for _, entry := range entries {
		var d float64
		// If PQ codebooks exist, use PQ reconstruction for approximate distance.
		if pq.codebooks != nil && len(entry.Codes) == pq.numSubquantizers {
			approxResidual, err := pq.decodePQCode(entry.Codes)
			if err != nil {
				d = pq.Distance(query, entry.Vector)
			} else {
				approxVec, err := vectorAdd(pq.coarseCentroids[entry.Cluster], approxResidual)
				if err != nil {
					d = pq.Distance(query, entry.Vector)
				} else {
					d = pq.Distance(query, approxVec)
				}
			}
		} else {
			d = pq.Distance(query, entry.Vector)
		}
		results = append(results, core.Neighbor{ID: entry.ID, Distance: d})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	if k > len(results) {
		k = len(results)
	}
	return results[:k], nil
}

// Stats returns statistics about the index (e.g. total number of entries).
func (pq *PQIVFIndex) Stats() core.IndexStats {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	count := 0
	for _, entries := range pq.invertedLists {
		count += len(entries)
	}
	return core.IndexStats{
		Count:     count,
		Dimension: pq.dimension,
		Distance:  "euclidean",
	}
}

// serializedPQIVF is a serializable representation of the PQIVF index.
type serializedPQIVF struct {
	Dimension        int
	CoarseK          int
	CoarseCentroids  [][]float32
	ClusterCounts    map[int]int
	InvertedLists    map[int][]pqEntry
	NumSubquantizers int
	Codebooks        [][][]float32
	PqK              int
	KMeansIters      int
}

// GobEncode serializes the index into bytes using gob.
func (pq *PQIVFIndex) GobEncode() ([]byte, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	ser := serializedPQIVF{
		Dimension:        pq.dimension,
		CoarseK:          pq.coarseK,
		CoarseCentroids:  pq.coarseCentroids,
		ClusterCounts:    pq.clusterCounts,
		InvertedLists:    pq.invertedLists,
		NumSubquantizers: pq.numSubquantizers,
		Codebooks:        pq.codebooks,
		PqK:              pq.pqK,
		KMeansIters:      pq.kMeansIters,
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(ser); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode deserializes the index from bytes using gob.
func (pq *PQIVFIndex) GobDecode(data []byte) error {
	var ser serializedPQIVF
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&ser); err != nil {
		return err
	}
	pq.dimension = ser.Dimension
	pq.coarseK = ser.CoarseK
	pq.coarseCentroids = ser.CoarseCentroids
	pq.clusterCounts = ser.ClusterCounts
	pq.invertedLists = ser.InvertedLists
	pq.numSubquantizers = ser.NumSubquantizers
	pq.codebooks = ser.Codebooks
	pq.pqK = ser.PqK
	pq.kMeansIters = ser.KMeansIters
	pq.idToCluster = make(map[int]int)
	// Rebuild idToCluster mapping from the inverted lists.
	for cluster, entries := range pq.invertedLists {
		for _, entry := range entries {
			pq.idToCluster[entry.ID] = cluster
		}
	}
	pq.Distance = core.Euclidean
	return nil
}

// Save writes the index to disk using gob encoding.
func (pq *PQIVFIndex) Save(path string) error {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	return enc.Encode(pq)
}

// Load reads the index from disk using gob decoding.
func (pq *PQIVFIndex) Load(path string) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	return dec.Decode(pq)
}

// Check interface compliance.
var _ core.Index = (*PQIVFIndex)(nil)

// init registers types for gob encoding.
func init() {
	gob.Register(&PQIVFIndex{})
	gob.Register(pqEntry{})
}

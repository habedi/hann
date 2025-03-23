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

// Internal pseudo-random number generator for the package.
var seededRand = rand.New(rand.NewSource(core.GetSeed()))

// pqEntry represents an entry stored in an inverted list.
type pqEntry struct {
	ID      int
	Vector  []float32
	Codes   []int // product quantization codes (one int per subquantizer)
	Cluster int   // which coarse cluster this vector belongs to
}

// PQIVFIndex implements a full Product Quantization Inverted File (IVF-PQ) index.
// It first does coarse quantization (online clustering) and then product–quantizes
// the residuals (the difference between a vector and its coarse centroid).
type PQIVFIndex struct {
	mu              sync.RWMutex
	dimension       int
	coarseK         int               // number of coarse clusters
	coarseCentroids [][]float32       // one centroid per cluster
	clusterCounts   map[int]int       // number of vectors assigned per cluster
	invertedLists   map[int][]pqEntry // maps cluster id to list of entries

	// Product quantization fields.
	numSubquantizers int           // number of subspaces for product quantization
	codebooks        [][][]float32 // global codebooks, one per subquantizer

	// Quantization parameters.
	pqK         int // number of codewords per subquantizer
	kMeansIters int // number of iterations for k-means training

	// Auxiliary map for quick duplicate check and deletion.
	idToCluster map[int]int // maps vector id to its assigned cluster

	// Distance metric fields.
	Distance     core.DistanceFunc // internal distance function
	DistanceName string            // human–readable name for the distance metric
}

// NewPQIVFIndex creates a new PQIVF index with the given parameters.
func NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters int, distance core.DistanceFunc, distanceName string) *PQIVFIndex {
	return &PQIVFIndex{
		dimension:        dimension,
		coarseK:          coarseK,
		coarseCentroids:  make([][]float32, 0),
		clusterCounts:    make(map[int]int),
		invertedLists:    make(map[int][]pqEntry),
		numSubquantizers: numSubquantizers,
		codebooks:        nil, // not trained yet
		pqK:              pqK,
		kMeansIters:      kMeansIters,
		idToCluster:      make(map[int]int),
		Distance:         distance,
		DistanceName:     distanceName,
	}
}

// ----- Coarse Quantization (online clustering) -----

// nearestCentroid finds the index of the nearest coarse centroid to the given vector.
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

// nearestCentroids returns a sorted slice of candidate clusters (with their distances)
// in ascending order of distance to the given vector.
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

// ----- Add / BulkAdd -----

// Add inserts a vector with a given id into the index.
func (pq *PQIVFIndex) Add(id int, vector []float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(vector) != pq.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), pq.dimension)
	}
	if _, exists := pq.idToCluster[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}
	// Normalize the vector if using cosine distance.
	if pq.DistanceName == "cosine" {
		core.NormalizeVector(vector)
	}

	var cluster int
	if len(pq.coarseCentroids) < pq.coarseK {
		cluster = len(pq.coarseCentroids)
		centroid := make([]float32, pq.dimension)
		copy(centroid, vector)
		pq.coarseCentroids = append(pq.coarseCentroids, centroid)
		pq.clusterCounts[cluster] = 1
	} else {
		cluster, _ = pq.nearestCentroid(vector)
		n := pq.clusterCounts[cluster]
		centroid := pq.coarseCentroids[cluster]
		for i := 0; i < pq.dimension; i++ {
			centroid[i] = (centroid[i]*float32(n) + vector[i]) / float32(n+1)
		}
		pq.clusterCounts[cluster] = n + 1
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
	return nil
}

// BulkAdd inserts multiple vectors into the index.
func (pq *PQIVFIndex) BulkAdd(vectors map[int][]float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	// If using cosine distance, perform bulk normalization.
	if pq.DistanceName == "cosine" {
		var vecs [][]float32
		for id, vector := range vectors {
			if len(vector) != pq.dimension {
				return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), pq.dimension, id)
			}
			vecs = append(vecs, vector)
		}
		core.NormalizeBatch(vecs)
	}

	for id, vector := range vectors {
		if len(vector) != pq.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), pq.dimension, id)
		}
		if _, exists := pq.idToCluster[id]; exists {
			return fmt.Errorf("id %d already exists", id)
		}

		var cluster int
		if len(pq.coarseCentroids) < pq.coarseK {
			cluster = len(pq.coarseCentroids)
			centroid := make([]float32, pq.dimension)
			copy(centroid, vector)
			pq.coarseCentroids = append(pq.coarseCentroids, centroid)
			pq.clusterCounts[cluster] = 1
		} else {
			cluster, _ = pq.nearestCentroid(vector)
			n := pq.clusterCounts[cluster]
			centroid := pq.coarseCentroids[cluster]
			for i := 0; i < pq.dimension; i++ {
				centroid[i] = (centroid[i]*float32(n) + vector[i]) / float32(n+1)
			}
			pq.clusterCounts[cluster] = n + 1
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
	}
	return nil
}

// Delete removes the vector with the given id from the index.
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
	for i, entry := range entries {
		if entry.ID == id {
			pq.invertedLists[cluster] = append(entries[:i], entries[i+1:]...)
			found = true
			pq.clusterCounts[cluster]--
			break
		}
	}
	if !found {
		return fmt.Errorf("id %d not found in cluster %d", id, cluster)
	}
	delete(pq.idToCluster, id)
	return nil
}

// BulkDelete removes multiple vectors from the index.
func (pq *PQIVFIndex) BulkDelete(ids []int) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	for _, id := range ids {
		cluster, exists := pq.idToCluster[id]
		if !exists {
			continue
		}
		entries, ok := pq.invertedLists[cluster]
		if !ok {
			continue
		}
		for i, entry := range entries {
			if entry.ID == id {
				pq.invertedLists[cluster] = append(entries[:i], entries[i+1:]...)
				pq.clusterCounts[cluster]--
				break
			}
		}
		delete(pq.idToCluster, id)
	}
	return nil
}

// Update modifies the vector associated with the given id.
// For simplicity, we remove the old entry and re-add the new one.
func (pq *PQIVFIndex) Update(id int, vector []float32) error {
	if err := pq.Delete(id); err != nil {
		return err
	}
	return pq.Add(id, vector)
}

// BulkUpdate updates multiple vectors in the index.
func (pq *PQIVFIndex) BulkUpdate(updates map[int][]float32) error {
	for id, vector := range updates {
		if err := pq.Update(id, vector); err != nil {
			return err
		}
	}
	return nil
}

// ----- Training for Product Quantization -----

func (pq *PQIVFIndex) Train() error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(pq.invertedLists) == 0 {
		return fmt.Errorf("no data to train on")
	}

	dataPerSub := make([][][]float32, pq.numSubquantizers)
	for i := 0; i < pq.numSubquantizers; i++ {
		dataPerSub[i] = make([][]float32, 0)
	}

	for cluster, entries := range pq.invertedLists {
		centroid := pq.coarseCentroids[cluster]
		for _, entry := range entries {
			residual := vectorSub(entry.Vector, centroid)
			subVecs := splitVector(residual, pq.numSubquantizers)
			for i, sub := range subVecs {
				dataPerSub[i] = append(dataPerSub[i], sub)
			}
		}
	}

	codebooks := make([][][]float32, pq.numSubquantizers)
	for i := 0; i < pq.numSubquantizers; i++ {
		cb, err := trainSubquantizer(dataPerSub[i], pq.pqK, pq.kMeansIters)
		if err != nil {
			return err
		}
		codebooks[i] = cb
	}
	pq.codebooks = codebooks

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

func (pq *PQIVFIndex) encodeVector(vector []float32, cluster int) ([]int, error) {
	if pq.codebooks == nil {
		return nil, fmt.Errorf("codebooks not trained")
	}
	residual := vectorSub(vector, pq.coarseCentroids[cluster])
	subVecs := splitVector(residual, pq.numSubquantizers)
	codes := make([]int, pq.numSubquantizers)
	for i, sub := range subVecs {
		best := -1
		bestDist := math.MaxFloat64
		for j, cent := range pq.codebooks[i] {
			d := euclidean(sub, cent)
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

// ----- Helper Functions for Vector Operations -----

func vectorSub(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("vector lengths do not match")
	}
	res := make([]float32, len(a))
	for i := range a {
		res[i] = a[i] - b[i]
	}
	return res
}

func vectorAdd(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("vector lengths do not match")
	}
	res := make([]float32, len(a))
	for i := range a {
		res[i] = a[i] + b[i]
	}
	return res
}

func splitVector(vec []float32, numParts int) [][]float32 {
	total := len(vec)
	subDim := total / numParts
	remainder := total % numParts
	parts := make([][]float32, numParts)
	start := 0
	for i := 0; i < numParts; i++ {
		end := start + subDim
		if i == numParts-1 {
			end += remainder
		}
		parts[i] = vec[start:end]
		start = end
	}
	return parts
}

func euclidean(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("vector lengths do not match")
	}
	sum := 0.0
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func trainSubquantizer(data [][]float32, k int, iterations int) ([][]float32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("no data for subquantizer training")
	}
	if len(data) < k {
		k = len(data)
	}
	centroids := make([][]float32, k)
	perm := seededRand.Perm(len(data))
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
				d := euclidean(point, cent)
				if d < bestDist {
					bestDist = d
					best = i
				}
			}
			clusters[best] = append(clusters[best], point)
		}
		for i, cluster := range clusters {
			if len(cluster) == 0 {
				continue
			}
			newCentroid := make([]float32, len(data[0]))
			for _, point := range cluster {
				for j, v := range point {
					newCentroid[j] += v
				}
			}
			for j := range newCentroid {
				newCentroid[j] /= float32(len(cluster))
			}
			centroids[i] = newCentroid
		}
	}
	return centroids, nil
}

// ----- Search -----

// Search returns the k nearest neighbors for a query vector.
func (pq *PQIVFIndex) Search(query []float32, k int) ([]core.Neighbor, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if len(query) != pq.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), pq.dimension)
	}
	// Normalize the query if using cosine distance.
	queryCopy := make([]float32, len(query))
	copy(queryCopy, query)
	if pq.DistanceName == "cosine" {
		core.NormalizeVector(queryCopy)
	}
	query = queryCopy

	if len(pq.invertedLists) == 0 {
		return nil, fmt.Errorf("index is empty")
	}

	candidates := pq.nearestCentroids(query)
	numCandidates := 3
	if numCandidates > len(candidates) {
		numCandidates = len(candidates)
	}
	var entries []pqEntry
	for i := 0; i < numCandidates; i++ {
		cluster := candidates[i].cluster
		entries = append(entries, pq.invertedLists[cluster]...)
	}
	if len(entries) < k {
		var allEntries []pqEntry
		for _, list := range pq.invertedLists {
			allEntries = append(allEntries, list...)
		}
		entries = allEntries
	}

	var results []core.Neighbor
	for _, entry := range entries {
		var d float64
		if pq.codebooks != nil && len(entry.Codes) == pq.numSubquantizers {
			approxResidual, err := pq.decodePQCode(entry.Codes)
			if err != nil {
				d = pq.Distance(query, entry.Vector)
			} else {
				approxVec := vectorAdd(pq.coarseCentroids[entry.Cluster], approxResidual)
				d = pq.Distance(query, approxVec)
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

// Stats returns metadata about the index.
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
		Distance:  pq.DistanceName,
	}
}

// ----- Persistence -----

type serializedPQIVF struct {
	Dimension        int
	CoarseK          int
	CoarseCentroids  [][]float32
	ClusterCounts    map[int]int
	InvertedLists    map[int][]pqEntry
	NumSubquantizers int
	Codebooks        [][][]float32
	DistanceName     string
	PqK              int
	KMeansIters      int
}

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
		DistanceName:     pq.DistanceName,
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
	pq.DistanceName = ser.DistanceName
	pq.pqK = ser.PqK
	pq.kMeansIters = ser.KMeansIters
	pq.idToCluster = make(map[int]int)
	for cluster, entries := range pq.invertedLists {
		for _, entry := range entries {
			pq.idToCluster[entry.ID] = cluster
		}
	}
	return nil
}

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

var _ core.Index = (*PQIVFIndex)(nil)

func init() {
	gob.Register(&PQIVFIndex{})
	gob.Register(pqEntry{})
}

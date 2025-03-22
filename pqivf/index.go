package pqivf

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"

	"github.com/habedi/hann/core"
)

// pqEntry represents an entry stored in an inverted list.
// For simplicity, we store the full vector along with the identifier.
type pqEntry struct {
	ID     int
	Vector []float32
}

// PQIVFIndex implements a simplified Product Quantization Inverted File (PQIVF) index.
// This implementation performs basic online clustering for coarse quantization.
// (Note: full product quantization is not implemented.)
type PQIVFIndex struct {
	mu              sync.RWMutex
	dimension       int
	coarseK         int               // number of coarse clusters
	coarseCentroids [][]float32       // one centroid per cluster
	clusterCounts   map[int]int       // number of vectors assigned per cluster
	invertedLists   map[int][]pqEntry // maps cluster id to list of entries

	// For a full PQIVF, the following fields would be used:
	numSubquantizers int           // number of subspaces for product quantization
	codebooks        [][][]float32 // per-subspace codebooks (not implemented here)

	// Auxiliary map for quick duplicate check and deletion:
	idToCluster map[int]int // maps vector id to its assigned cluster

	// New fields for distance metric support.
	Distance     core.DistanceFunc // internal distance function
	DistanceName string            // human–readable name for the distance metric
}

// NewPQIVFIndex creates a new PQIVF index with the given parameters,
// including the distance function and its name.
func NewPQIVFIndex(dimension, coarseK, numSubquantizers int, distance core.DistanceFunc, distanceName string) *PQIVFIndex {
	return &PQIVFIndex{
		dimension:        dimension,
		coarseK:          coarseK,
		coarseCentroids:  make([][]float32, 0),
		clusterCounts:    make(map[int]int),
		invertedLists:    make(map[int][]pqEntry),
		numSubquantizers: numSubquantizers,
		codebooks:        nil, // not implemented
		idToCluster:      make(map[int]int),
		Distance:         distance,
		DistanceName:     distanceName,
	}
}

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

// Add inserts a vector with a given id into the index.
// It assigns the vector to the nearest coarse centroid; if fewer than coarseK clusters
// exist, a new cluster is created.
func (pq *PQIVFIndex) Add(id int, vector []float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(vector) != pq.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), pq.dimension)
	}
	// Check if id already exists.
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
	entry := pqEntry{ID: id, Vector: vector}
	pq.invertedLists[cluster] = append(pq.invertedLists[cluster], entry)
	return nil
}

// BulkAdd inserts multiple vectors into the index.
func (pq *PQIVFIndex) BulkAdd(vectors map[int][]float32) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

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
		entry := pqEntry{ID: id, Vector: vector}
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

// Search returns the k nearest neighbors (ids and distances) for a query vector.
// It uses the internal distance function.
func (pq *PQIVFIndex) Search(query []float32, k int) ([]core.Neighbor, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if len(query) != pq.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), pq.dimension)
	}
	if len(pq.invertedLists) == 0 {
		return nil, fmt.Errorf("index is empty")
	}

	// Get the top candidate clusters (up to 3 or fewer if there are fewer clusters).
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
	// Fallback: if not enough entries, search across all clusters.
	if len(entries) < k {
		var allEntries []pqEntry
		for _, list := range pq.invertedLists {
			allEntries = append(allEntries, list...)
		}
		entries = allEntries
	}

	var results []core.Neighbor
	for _, entry := range entries {
		d := pq.Distance(query, entry.Vector)
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

// Stats returns metadata about the index, including the chosen distance metric.
func (pq *PQIVFIndex) Stats() core.IndexStats {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	count := 0
	for _, entries := range pq.invertedLists {
		count += len(entries)
	}
	size := count * pq.dimension * 4 // approximate size in bytes
	return core.IndexStats{
		Count:     count,
		Dimension: pq.dimension,
		Size:      size,
		Distance:  pq.DistanceName,
	}
}

// --- Persistence ---
//
// Custom gob encoding to persist all fields of the index.
// The auxiliary idToCluster map is not persisted (it is rebuilt during decoding).

type serializedPQIVF struct {
	Dimension        int
	CoarseK          int
	CoarseCentroids  [][]float32
	ClusterCounts    map[int]int
	InvertedLists    map[int][]pqEntry
	NumSubquantizers int
	Codebooks        [][][]float32
	DistanceName     string
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
	// Rebuild the auxiliary map.
	pq.idToCluster = make(map[int]int)
	for cluster, entries := range pq.invertedLists {
		for _, entry := range entries {
			pq.idToCluster[entry.ID] = cluster
		}
	}
	return nil
}

// Save persists the index state to the specified file.
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

// Load initializes the index from a previously saved state.
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

// Ensure PQIVFIndex implements the core.Index interface.
var _ core.Index = (*PQIVFIndex)(nil)

func init() {
	gob.Register(&PQIVFIndex{})
	gob.Register(pqEntry{})
}

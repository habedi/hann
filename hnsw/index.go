package hnsw

import (
	"container/heap"
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/habedi/hann/core"
	"github.com/rs/zerolog/log"
)

// candidate is used for internal candidate management.
type candidate struct {
	node *Node
	dist float64
}

// candidateMinHeap implements a min–heap (the lowest distance at top).
type candidateMinHeap []candidate

func (h candidateMinHeap) Len() int            { return len(h) }
func (h candidateMinHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h candidateMinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *candidateMinHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *candidateMinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// candidateMaxHeap implements a max–heap (largest distance at top).
type candidateMaxHeap []candidate

func (h candidateMaxHeap) Len() int            { return len(h) }
func (h candidateMaxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h candidateMaxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *candidateMaxHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *candidateMaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// Node represents an element in the HNSW graph.
type Node struct {
	ID     int
	Vector []float32
	Level  int
	// Links maps each level to a slice of neighbor nodes.
	Links map[int][]*Node
	// reverseLinks tracks which nodes point to this node (for efficient deletion)
	reverseLinks map[int][]*Node
}

// HNSWIndex implements a simplified HNSW index.
type HNSWIndex struct {
	mu         sync.RWMutex
	dimension  int
	entryPoint *Node
	maxLevel   int
	nodes      map[int]*Node

	// Parameters (simplified)
	M  int // maximum number of neighbors per node
	ef int // search parameter (candidate list size for search)
}

func init() {
	rand.Seed(time.Now().UnixNano())
	log.Debug().Msg("Initialized random seed for HNSW index")
}

// NewHNSW creates a new HNSW index with the given dimension and parameters.
func NewHNSW(dimension int, M int, ef int) *HNSWIndex {
	log.Info().Msgf("Creating new HNSW index with dimension=%d, M=%d, ef=%d", dimension, M, ef)
	return &HNSWIndex{
		dimension: dimension,
		nodes:     make(map[int]*Node),
		maxLevel:  -1,
		M:         M,
		ef:        ef,
	}
}

// randomLevel generates a random level for a new node using an exponential distribution.
func (h *HNSWIndex) randomLevel() int {
	if h.M <= 1 {
		return 0
	}
	level := int(-math.Log(rand.Float64()) / math.Log(float64(h.M)))
	log.Debug().Msgf("Generated random level %d", level)
	return level
}

// euclidean is a simple Euclidean distance function.
func euclidean(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

// selectM selects up to M nearest candidates from the given candidate slice.
func selectM(candidates []candidate, M int) []candidate {
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})
	if len(candidates) > M {
		return candidates[:M]
	}
	return candidates
}

// min returns the smaller of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// selectNodes sorts a slice of nodes by their distance to vec and returns the top M nodes.
func selectNodes(nodes []*Node, vec []float32, M int) []*Node {
	sort.Slice(nodes, func(i, j int) bool {
		return euclidean(vec, nodes[i].Vector) < euclidean(vec, nodes[j].Vector)
	})
	if len(nodes) > M {
		return nodes[:M]
	}
	return nodes
}

// removeFromSlice removes a given node from a slice.
func removeFromSlice(slice []*Node, target *Node) []*Node {
	newSlice := slice[:0]
	for _, n := range slice {
		if n != target {
			newSlice = append(newSlice, n)
		}
	}
	return newSlice
}

// removeNodeLinks removes all references to node n from its neighbors.
func (h *HNSWIndex) removeNodeLinks(n *Node) {
	// Remove n from each neighbor's Links (tracked via reverseLinks)
	for level, neighbors := range n.reverseLinks {
		for _, neighbor := range neighbors {
			neighbor.Links[level] = removeFromSlice(neighbor.Links[level], n)
		}
		n.reverseLinks[level] = nil
	}
	// Also remove n from neighbor.reverseLinks for nodes that n points to.
	for level, neighbors := range n.Links {
		for _, neighbor := range neighbors {
			neighbor.reverseLinks[level] = removeFromSlice(neighbor.reverseLinks[level], n)
		}
		n.Links[level] = nil
	}
}

// insertNode performs the insertion of a node into the graph.
// The searchEf parameter controls the candidate list size used during linking.
func (h *HNSWIndex) insertNode(n *Node, searchEf int) {
	// If index is empty, set new node as entry point.
	if h.entryPoint == nil {
		h.entryPoint = n
		h.maxLevel = n.Level
		return
	}
	if n.Level > h.maxLevel {
		h.entryPoint = n
		h.maxLevel = n.Level
	}
	current := h.entryPoint
	// Greedy descent from the top level down to (n.Level+1)
	for L := h.maxLevel; L > n.Level; L-- {
		changed := true
		for changed {
			changed = false
			for _, neighbor := range current.Links[L] {
				if euclidean(n.Vector, neighbor.Vector) < euclidean(n.Vector, current.Vector) {
					current = neighbor
					changed = true
				}
			}
		}
	}
	// For levels from min(n.Level, h.maxLevel) down to 0, perform localized search.
	for L := min(n.Level, h.maxLevel); L >= 0; L-- {
		candList := h.searchLayer(n.Vector, current, L, searchEf, euclidean)
		selectedCands := selectM(candList, h.M)
		selectedNodes := make([]*Node, len(selectedCands))
		for i, cand := range selectedCands {
			selectedNodes[i] = cand.node
		}
		n.Links[L] = selectedNodes

		// Link back: add n to each selected neighbor's links and update reverseLinks.
		for _, neighbor := range selectedNodes {
			neighbor.Links[L] = append(neighbor.Links[L], n)
			neighbor.reverseLinks[L] = append(neighbor.reverseLinks[L], n)
			if len(neighbor.Links[L]) > h.M {
				neighbor.Links[L] = selectNodes(neighbor.Links[L], neighbor.Vector, h.M)
			}
		}

		if len(candList) > 0 {
			current = candList[0].node
		}
	}
}

// searchLayer performs a best–first search at the given level using a two–heap approach.
func (h *HNSWIndex) searchLayer(query []float32, entrypoint *Node, level int, ef int, distance func([]float32, []float32) float64) []candidate {
	visited := map[int]bool{entrypoint.ID: true}
	d0 := distance(query, entrypoint.Vector)

	// Use a min–heap for candidates to expand and a max–heap for results.
	candQueue := candidateMinHeap{{entrypoint, d0}}
	heap.Init(&candQueue)
	resultQueue := candidateMaxHeap{{entrypoint, d0}}
	heap.Init(&resultQueue)

	for candQueue.Len() > 0 {
		current := candQueue[0]       // candidate with smallest distance
		worstResult := resultQueue[0] // candidate with largest distance among current results
		if current.dist > worstResult.dist {
			break
		}
		heap.Pop(&candQueue) // expand current candidate

		// Expand neighbors
		for _, neighbor := range current.node.Links[level] {
			if visited[neighbor.ID] {
				continue
			}
			visited[neighbor.ID] = true
			d := distance(query, neighbor.Vector)
			if resultQueue.Len() < ef || d < resultQueue[0].dist {
				newCand := candidate{neighbor, d}
				heap.Push(&candQueue, newCand)
				heap.Push(&resultQueue, newCand)
				if resultQueue.Len() > ef {
					heap.Pop(&resultQueue)
				}
			}
		}
	}

	// Extract results from resultQueue and sort them in ascending order.
	results := make([]candidate, resultQueue.Len())
	for i := range results {
		results[i] = heap.Pop(&resultQueue).(candidate)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})
	return results
}

// Add inserts a vector with a given id into the HNSW index.
func (h *HNSWIndex) Add(id int, vector []float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if len(vector) != h.dimension {
		err := fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), h.dimension)
		log.Error().Err(err).Msg("Add failed due to dimension mismatch")
		return err
	}
	if _, exists := h.nodes[id]; exists {
		err := fmt.Errorf("id %d already exists", id)
		log.Error().Err(err).Msg("Add failed: duplicate id")
		return err
	}

	level := h.randomLevel()
	newNode := &Node{
		ID:           id,
		Vector:       vector,
		Level:        level,
		Links:        make(map[int][]*Node),
		reverseLinks: make(map[int][]*Node),
	}
	h.nodes[id] = newNode
	log.Debug().Msgf("Added node %d at level %d", id, level)

	h.insertNode(newNode, h.ef)
	return nil
}

// Delete removes the vector with the given id from the index.
func (h *HNSWIndex) Delete(id int) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	node, exists := h.nodes[id]
	if !exists {
		err := fmt.Errorf("id %d not found", id)
		log.Error().Err(err).Msg("Delete failed")
		return err
	}
	h.removeNodeLinks(node)
	delete(h.nodes, id)
	log.Info().Msgf("Deleted node %d", id)

	// Update entry point if necessary.
	if h.entryPoint != nil && h.entryPoint.ID == id {
		h.entryPoint = nil
		for _, n := range h.nodes {
			h.entryPoint = n
			break
		}
		log.Info().Msg("Updated entry point after deletion")
	}
	return nil
}

// Update modifies the vector associated with the given id.
// This version re–links the updated node to keep the graph consistent.
func (h *HNSWIndex) Update(id int, vector []float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	node, exists := h.nodes[id]
	if !exists {
		err := fmt.Errorf("id %d not found", id)
		log.Error().Err(err).Msg("Update failed")
		return err
	}
	if len(vector) != h.dimension {
		err := fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), h.dimension)
		log.Error().Err(err).Msg("Update failed due to dimension mismatch")
		return err
	}
	// Remove stale links.
	h.removeNodeLinks(node)
	// Update the vector and recalculate the node's level.
	node.Vector = vector
	node.Level = h.randomLevel()
	node.Links = make(map[int][]*Node)
	node.reverseLinks = make(map[int][]*Node)
	h.insertNode(node, h.ef)
	log.Info().Msgf("Updated node %d", id)
	return nil
}

// BulkAdd inserts multiple vectors into the index in a single lock acquisition.
// Improvements applied:
//   - Precompute nodes and sort by descending level.
//   - Temporarily lower ef during bulk insertion.
func (h *HNSWIndex) BulkAdd(vectors map[int][]float32) error {
	// Preprocess nodes outside the lock.
	nodesSlice := make([]*Node, 0, len(vectors))
	for id, vector := range vectors {
		if len(vector) != h.dimension {
			err := fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), h.dimension, id)
			log.Error().Err(err).Msg("BulkAdd failed due to dimension mismatch")
			return err
		}
		if _, exists := h.nodes[id]; exists {
			err := fmt.Errorf("id %d already exists", id)
			log.Error().Err(err).Msg("BulkAdd failed: duplicate id")
			return err
		}

		level := h.randomLevel()
		newNode := &Node{
			ID:           id,
			Vector:       vector,
			Level:        level,
			Links:        make(map[int][]*Node),
			reverseLinks: make(map[int][]*Node),
		}
		nodesSlice = append(nodesSlice, newNode)
	}

	// Sort nodes by descending level so that high–level nodes are inserted first.
	sort.Slice(nodesSlice, func(i, j int) bool {
		return nodesSlice[i].Level > nodesSlice[j].Level
	})

	// Temporarily lower ef during bulk insertion to speed up linking.
	bulkEf := h.ef
	if bulkEf > 16 {
		bulkEf = 16
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Insert nodes in sorted order.
	for _, newNode := range nodesSlice {
		h.nodes[newNode.ID] = newNode
		if h.entryPoint == nil {
			h.entryPoint = newNode
			h.maxLevel = newNode.Level
		} else {
			if newNode.Level > h.maxLevel {
				h.entryPoint = newNode
				h.maxLevel = newNode.Level
			}
			h.insertNode(newNode, bulkEf)
		}
		log.Debug().Msgf("BulkAdd: inserted node %d at level %d", newNode.ID, newNode.Level)
	}
	log.Info().Msgf("BulkAdd: added %d vectors", len(vectors))
	return nil
}

// BulkDelete removes multiple vectors from the index given a slice of ids.
func (h *HNSWIndex) BulkDelete(ids []int) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	for _, id := range ids {
		if _, exists := h.nodes[id]; !exists {
			log.Warn().Msgf("BulkDelete: id %d not found; skipping", id)
			continue
		}
		// Remove id from all neighbor links in all nodes.
		for _, n := range h.nodes {
			for L, neighbors := range n.Links {
				newNeighbors := make([]*Node, 0, len(neighbors))
				for _, neighbor := range neighbors {
					if neighbor.ID != id {
						newNeighbors = append(newNeighbors, neighbor)
					}
				}
				n.Links[L] = newNeighbors
			}
		}
		delete(h.nodes, id)
		log.Info().Msgf("Deleted node %d", id)
		if h.entryPoint != nil && h.entryPoint.ID == id {
			h.entryPoint = nil
			for _, n := range h.nodes {
				h.entryPoint = n
				break
			}
			log.Info().Msg("Updated entry point after deletion")
		}
	}

	// Cleanup pass: remove any links that point to deleted nodes.
	for _, n := range h.nodes {
		for L, neighbors := range n.Links {
			newNeighbors := make([]*Node, 0, len(neighbors))
			for _, neighbor := range neighbors {
				if _, exists := h.nodes[neighbor.ID]; exists {
					newNeighbors = append(newNeighbors, neighbor)
				}
			}
			n.Links[L] = newNeighbors
		}
	}
	log.Info().Msgf("BulkDelete: removed %d vectors", len(ids))
	return nil
}

// BulkUpdate updates multiple vectors in the index.
// 'updates' is a map from id to the new vector.
func (h *HNSWIndex) BulkUpdate(updates map[int][]float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	for id, vector := range updates {
		node, exists := h.nodes[id]
		if !exists {
			log.Warn().Msgf("BulkUpdate: id %d not found; skipping", id)
			continue
		}
		if len(vector) != h.dimension {
			err := fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), h.dimension, id)
			log.Error().Err(err).Msg("BulkUpdate failed")
			return err
		}
		h.removeNodeLinks(node)
		node.Vector = vector
		node.Level = h.randomLevel()
		node.Links = make(map[int][]*Node)
		node.reverseLinks = make(map[int][]*Node)
		h.insertNode(node, h.ef)
	}
	log.Info().Msgf("BulkUpdate: updated %d vectors", len(updates))
	return nil
}

// Search returns the ids and distances of the k nearest neighbors for a query vector.
func (h *HNSWIndex) Search(query []float32, k int, distance core.DistanceFunc) ([]core.Neighbor, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(query) != h.dimension {
		err := fmt.Errorf("query dimension %d does not match index dimension %d", len(query), h.dimension)
		log.Error().Err(err).Msg("Search failed")
		return nil, err
	}
	if h.entryPoint == nil {
		err := errors.New("index is empty")
		log.Error().Err(err).Msg("Search failed")
		return nil, err
	}

	log.Debug().Msg("Starting greedy search from entry point")
	current := h.entryPoint
	for L := h.maxLevel; L > 0; L-- {
		changed := true
		for changed {
			changed = false
			for _, neighbor := range current.Links[L] {
				if distance(query, neighbor.Vector) < distance(query, current.Vector) {
					current = neighbor
					changed = true
				}
			}
		}
	}

	// Use the candidate heap search at level 0.
	candidates := h.searchLayer(query, current, 0, h.ef, distance)
	// Fallback: if fewer than k candidates, scan the entire index.
	if len(candidates) < k {
		candidateIDs := make(map[int]bool)
		for _, c := range candidates {
			candidateIDs[c.node.ID] = true
		}
		fallbackHeap := candidateMaxHeap{}
		heap.Init(&fallbackHeap)
		for _, node := range h.nodes {
			if candidateIDs[node.ID] {
				continue
			}
			d := distance(query, node.Vector)
			cand := candidate{node, d}
			if fallbackHeap.Len() < (k - len(candidates)) {
				heap.Push(&fallbackHeap, cand)
			} else if fallbackHeap.Len() > 0 && d < fallbackHeap[0].dist {
				heap.Pop(&fallbackHeap)
				heap.Push(&fallbackHeap, cand)
			}
		}
		fallbackCandidates := make([]candidate, fallbackHeap.Len())
		for i := range fallbackCandidates {
			fallbackCandidates[i] = heap.Pop(&fallbackHeap).(candidate)
		}
		candidates = append(candidates, fallbackCandidates...)
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].dist < candidates[j].dist
		})
	}

	if k > len(candidates) {
		k = len(candidates)
	}
	results := make([]core.Neighbor, k)
	for i := 0; i < k; i++ {
		results[i] = core.Neighbor{ID: candidates[i].node.ID, Distance: candidates[i].dist}
	}
	log.Debug().Msgf("Search completed: returning %d neighbors", k)
	return results, nil
}

// Stats returns metadata about the index.
func (h *HNSWIndex) Stats() core.IndexStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	count := len(h.nodes)
	size := count * h.dimension * 4 // rough estimate: each vector is dimension*4 bytes
	stats := core.IndexStats{
		Count:     count,
		Dimension: h.dimension,
		Size:      size,
	}
	log.Debug().Msgf("Index stats: %d vectors, %d dimensions, ~%d bytes", stats.Count, stats.Dimension, stats.Size)
	return stats
}

// Save persists the index state to the specified file.
func (h *HNSWIndex) Save(path string) error {
	h.mu.RLock()
	defer h.mu.RUnlock()
	f, err := os.Create(path)
	if err != nil {
		log.Error().Err(err).Msg("Save failed: could not create file")
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if err := enc.Encode(h); err != nil {
		log.Error().Err(err).Msg("Save failed during encoding")
		return err
	}
	log.Info().Msgf("Index saved to %s", path)
	return nil
}

// Load initializes the index from a previously saved state.
func (h *HNSWIndex) Load(path string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	f, err := os.Open(path)
	if err != nil {
		log.Error().Err(err).Msg("Load failed: could not open file")
		return err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	if err := dec.Decode(h); err != nil {
		log.Error().Err(err).Msg("Load failed during decoding")
		return err
	}
	log.Info().Msgf("Index loaded from %s", path)
	return nil
}

// --- Custom Gob Serialization ---
//
// Because Node.Links is cyclic (nodes reference each other),
// we serialize the index into an intermediate structure that stores neighbor IDs only.

type serializedNode struct {
	ID     int
	Vector []float32
	Level  int
	Links  map[int][]int // For each level, list of neighbor IDs.
}

type serializedHNSW struct {
	Dimension  int
	EntryPoint int // entry point ID; -1 if nil.
	MaxLevel   int
	Nodes      map[int]serializedNode
}

// Ensure HNSWIndex implements the core.Index interface.
var _ core.Index = (*HNSWIndex)(nil)

func init() {
	gob.Register(&HNSWIndex{})
	gob.Register(&Node{})
	log.Debug().Msg("Registered HNSWIndex and Node types for Gob encoding")
}

package hnsw

import (
	"bytes"
	"container/heap"
	"encoding/gob"
	"errors"
	"fmt"
	"github.com/habedi/hann/core"
	"github.com/rs/zerolog/log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
)

// seededRand is a global random number generator used for level generation.
var seededRand = rand.New(rand.NewSource(core.GetSeed()))
var seededRandMu sync.Mutex

// maxLevelCap is the upper bound for a node's level.
const maxLevelCap = 32

// candidate represents a potential neighbor with its distance.
type candidate struct {
	node *Node   // reference to the candidate node
	dist float64 // distance to the query vector
}

// candidateMinHeap implements a min-heap for candidates based on their distance.
type candidateMinHeap []candidate

func (h candidateMinHeap) Len() int { return len(h) }
func (h candidateMinHeap) Less(i, j int) bool {
	if h[i].dist == h[j].dist {
		return h[i].node.ID < h[j].node.ID
	}
	return h[i].dist < h[j].dist
}
func (h candidateMinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *candidateMinHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *candidateMinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// candidateMaxHeap implements a max-heap for candidates based on their distance.
type candidateMaxHeap []candidate

func (h candidateMaxHeap) Len() int { return len(h) }
func (h candidateMaxHeap) Less(i, j int) bool {
	if h[i].dist == h[j].dist {
		return h[i].node.ID < h[j].node.ID
	}
	return h[i].dist > h[j].dist
}
func (h candidateMaxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *candidateMaxHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *candidateMaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// Node represents a vector in the HNSW graph along with its links.
type Node struct {
	ID           int             // unique identifier of the node
	Vector       []float32       // vector data
	Level        int             // node level in the hierarchy
	Links        map[int][]*Node // links to neighbors at each level
	ReverseLinks map[int][]*Node // reverse links from neighbors
}

// HNSWIndex is the main structure for the HNSW graph index.
type HNSWIndex struct {
	Mu               sync.RWMutex      `gob:"-"` // mutex to control concurrent access
	Dimension        int               // dimension of the vectors
	EntryPoint       *Node             // starting point for searches
	MaxLevel         int               // current maximum level in the graph
	Nodes            map[int]*Node     // map of node id to Node pointer
	M                int               // maximum number of neighbors per node
	Ef               int               // search parameter controlling search depth
	Distance         core.DistanceFunc // function to calculate distance between vectors
	DistanceName     string            // name of the distance metric
	ExhaustiveSearch bool              // flag for performing exhaustive search during searchLayer
}

// NewHNSW creates a new HNSW index given the dimension, M, ef, and distance function.
func NewHNSW(dimension int, M int, ef int, distance core.DistanceFunc, distanceName string) *HNSWIndex {
	log.Info().Msgf("Creating new HNSW index with dimension=%d, M=%d, ef=%d, distance=%s",
		dimension, M, ef, distanceName)
	return &HNSWIndex{
		Dimension:    dimension,
		Nodes:        make(map[int]*Node),
		MaxLevel:     -1,
		M:            M,
		Ef:           ef,
		Distance:     distance,
		DistanceName: distanceName,
	}
}

// randomLevel computes a random level for a new node based on an exponential distribution.
func (h *HNSWIndex) randomLevel() int {
	if h.M <= 1 {
		return 0
	}
	seededRandMu.Lock()
	r := seededRand.Float64()
	seededRandMu.Unlock()
	level := int(-math.Log(r) / math.Log(float64(h.M)))
	if level > maxLevelCap {
		level = maxLevelCap
	}
	log.Debug().Msgf("Generated random level %d", level)
	return level
}

// serializedNode is used to store a Node during gob encoding/decoding.
type serializedNode struct {
	ID     int           // node id
	Vector []float32     // vector data
	Level  int           // node level
	Links  map[int][]int // neighbor ids at each level
}

// serializedIndex is the serializable version of the HNSWIndex.
type serializedIndex struct {
	Dimension    int                    // dimension of the index
	M            int                    // maximum neighbors per node
	Ef           int                    // search parameter
	Nodes        map[int]serializedNode // serialized nodes
	EntryPoint   int                    // id of the entry point node
	MaxLevel     int                    // maximum level in the graph
	DistanceName string                 // name of the distance metric
}

// GobEncode serializes the HNSWIndex using the gob encoder.
func (h *HNSWIndex) GobEncode() ([]byte, error) {
	h.Mu.RLock()
	defer h.Mu.RUnlock()
	si := serializedIndex{
		Dimension:    h.Dimension,
		M:            h.M,
		Ef:           h.Ef,
		Nodes:        make(map[int]serializedNode),
		EntryPoint:   0,
		MaxLevel:     h.MaxLevel,
		DistanceName: h.DistanceName,
	}
	for id, node := range h.Nodes {
		sn := serializedNode{
			ID:     node.ID,
			Vector: node.Vector,
			Level:  node.Level,
			Links:  make(map[int][]int),
		}
		// Store neighbor ids for each level.
		for level, neighbors := range node.Links {
			for _, nb := range neighbors {
				sn.Links[level] = append(sn.Links[level], nb.ID)
			}
		}
		si.Nodes[id] = sn
	}
	if h.EntryPoint != nil {
		si.EntryPoint = h.EntryPoint.ID
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(si); err != nil {
		log.Error().Err(err).Msg("Failed to encode HNSWIndex")
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode deserializes data into the HNSWIndex using the gob decoder.
func (h *HNSWIndex) GobDecode(data []byte) error {
	var si serializedIndex
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&si); err != nil {
		log.Error().Err(err).Msg("Failed to decode HNSWIndex")
		return err
	}
	h.Dimension = si.Dimension
	h.M = si.M
	h.Ef = si.Ef
	h.MaxLevel = si.MaxLevel
	h.DistanceName = si.DistanceName
	h.Nodes = make(map[int]*Node)
	// Recreate nodes from the serialized data.
	for id, sn := range si.Nodes {
		h.Nodes[id] = &Node{
			ID:           sn.ID,
			Vector:       sn.Vector,
			Level:        sn.Level,
			Links:        make(map[int][]*Node),
			ReverseLinks: make(map[int][]*Node),
		}
	}
	// Restore neighbor pointers.
	for id, sn := range si.Nodes {
		node := h.Nodes[id]
		for level, nbIDs := range sn.Links {
			for _, nbID := range nbIDs {
				if nb, exists := h.Nodes[nbID]; exists {
					node.Links[level] = append(node.Links[level], nb)
				}
			}
		}
	}
	// Rebuild reverse links.
	for _, node := range h.Nodes {
		for level, neighbors := range node.Links {
			for _, nb := range neighbors {
				nb.ReverseLinks[level] = append(nb.ReverseLinks[level], node)
			}
		}
	}
	if si.EntryPoint != 0 {
		h.EntryPoint = h.Nodes[si.EntryPoint]
	} else {
		h.EntryPoint = nil
	}
	return nil
}

// selectM chooses the top M candidates based on distance.
func selectM(candidates []candidate, M int) []candidate {
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].dist == candidates[j].dist {
			return candidates[i].node.ID < candidates[j].node.ID
		}
		return candidates[i].dist < candidates[j].dist
	})
	if len(candidates) > M {
		return candidates[:M]
	}
	return candidates
}

// selectNodes selects up to M nodes from a list based on their distance to vec.
func selectNodes(nodes []*Node, vec []float32, M int, distance func([]float32, []float32) float64) []*Node {
	// Create a temporary array with nodes and their distances.
	type nodeWithDist struct {
		node *Node
		dist float64
	}
	arr := make([]nodeWithDist, len(nodes))
	for i, n := range nodes {
		arr[i] = nodeWithDist{n, distance(vec, n.Vector)}
	}
	sort.Slice(arr, func(i, j int) bool {
		if arr[i].dist == arr[j].dist {
			return arr[i].node.ID < arr[j].node.ID
		}
		return arr[i].dist < arr[j].dist
	})
	selected := make([]*Node, minInt(len(arr), M))
	for i := range selected {
		selected[i] = arr[i].node
	}
	return selected
}

// removeFromSlice removes a target node from a slice of nodes.
func removeFromSlice(slice []*Node, target *Node) []*Node {
	newSlice := slice[:0]
	for _, n := range slice {
		if n != target {
			newSlice = append(newSlice, n)
		}
	}
	return newSlice
}

// difference returns nodes in a that are not in b.
func difference(a, b []*Node) []*Node {
	set := make(map[int]bool)
	for _, n := range b {
		set[n.ID] = true
	}
	var diff []*Node
	for _, n := range a {
		if !set[n.ID] {
			diff = append(diff, n)
		}
	}
	return diff
}

// trimNeighborLinks reduces a node's neighbors at a level to the best M based on distance.
func trimNeighborLinks(n *Node, level, M int, distance func([]float32, []float32) float64) {
	original := n.Links[level]
	trimmed := selectNodes(original, n.Vector, M, distance)
	removed := difference(original, trimmed)
	for _, r := range removed {
		r.ReverseLinks[level] = removeFromSlice(r.ReverseLinks[level], n)
	}
	n.Links[level] = trimmed
}

// removeNodeLinks removes all links of a node from the graph.
func (h *HNSWIndex) removeNodeLinks(n *Node) {
	for level, neighbors := range n.ReverseLinks {
		for _, neighbor := range neighbors {
			neighbor.Links[level] = removeFromSlice(neighbor.Links[level], n)
		}
		n.ReverseLinks[level] = nil
	}
	for level, neighbors := range n.Links {
		for _, neighbor := range neighbors {
			neighbor.ReverseLinks[level] = removeFromSlice(neighbor.ReverseLinks[level], n)
		}
		n.Links[level] = nil
	}
}

// minInt returns the smaller of two integers.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// insertNode adds a node into the HNSW graph, updating links as needed.
func (h *HNSWIndex) insertNode(n *Node, searchEf int) {
	// If index is empty, set this node as entry point.
	if h.EntryPoint == nil {
		h.EntryPoint = n
		h.MaxLevel = n.Level
		return
	}
	// Update entry point if the new node has a higher level.
	if n.Level > h.MaxLevel {
		h.EntryPoint = n
		h.MaxLevel = n.Level
	}
	current := h.EntryPoint
	// Navigate the graph from the top level down to the node's level.
	for L := h.MaxLevel; L > n.Level; L-- {
		changed := true
		for changed {
			changed = false
			for _, neighbor := range current.Links[L] {
				if h.Distance(n.Vector, neighbor.Vector) < h.Distance(n.Vector, current.Vector) {
					current = neighbor
					changed = true
				}
			}
		}
	}
	// For each level where the new node will be inserted.
	for L := minInt(n.Level, h.MaxLevel); L >= 0; L-- {
		candList := h.searchLayer(n.Vector, current, L, searchEf, h.Distance)
		selectedCands := selectM(candList, h.M)
		selectedNodes := make([]*Node, len(selectedCands))
		for i, cand := range selectedCands {
			selectedNodes[i] = cand.node
		}
		n.Links[L] = selectedNodes
		// Update neighbor links to include the new node.
		for _, neighbor := range selectedNodes {
			neighbor.Links[L] = append(neighbor.Links[L], n)
			neighbor.ReverseLinks[L] = append(neighbor.ReverseLinks[L], n)
			if len(neighbor.Links[L]) > h.M {
				trimNeighborLinks(neighbor, L, h.M, h.Distance)
			}
		}
		// Move the current pointer for the next level.
		if len(candList) > 0 {
			current = candList[0].node
		}
	}
}

// searchLayer performs a search in the graph at a given level.
func (h *HNSWIndex) searchLayer(query []float32, entrypoint *Node, level int, ef int, distance func([]float32, []float32) float64) []candidate {
	visited := map[int]bool{entrypoint.ID: true}
	d0 := distance(query, entrypoint.Vector)
	candQueue := candidateMinHeap{{entrypoint, d0}}
	heap.Init(&candQueue)
	resultQueue := candidateMaxHeap{{entrypoint, d0}}
	heap.Init(&resultQueue)
	// Explore candidates while there are promising ones.
	for candQueue.Len() > 0 {
		current := candQueue[0]
		worstResult := resultQueue[0]
		if current.dist > worstResult.dist && !h.ExhaustiveSearch {
			break
		}
		heap.Pop(&candQueue)
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
	// Collect and sort results.
	results := make([]candidate, resultQueue.Len())
	for i := range results {
		results[i] = heap.Pop(&resultQueue).(candidate)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].dist == results[j].dist {
			return results[i].node.ID < results[j].node.ID
		}
		return results[i].dist < results[j].dist
	})
	return results
}

// Add inserts a new vector into the index with a unique id.
func (h *HNSWIndex) Add(id int, vector []float32) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	if len(vector) != h.Dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(vector), h.Dimension)
	}
	// Normalize if using cosine similarity.
	if h.DistanceName == "cosine" {
		core.NormalizeVector(vector)
	}
	if _, exists := h.Nodes[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}
	level := h.randomLevel()
	newNode := &Node{
		ID:           id,
		Vector:       vector,
		Level:        level,
		Links:        make(map[int][]*Node),
		ReverseLinks: make(map[int][]*Node),
	}
	h.Nodes[id] = newNode
	h.insertNode(newNode, h.Ef)
	return nil
}

// Delete removes a vector from the index by its id.
func (h *HNSWIndex) Delete(id int) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	node, exists := h.Nodes[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	h.removeNodeLinks(node)
	delete(h.Nodes, id)
	// Update the entry point if necessary.
	if h.EntryPoint != nil && h.EntryPoint.ID == id {
		h.EntryPoint = nil
		for _, n := range h.Nodes {
			if h.EntryPoint == nil || n.Level > h.EntryPoint.Level {
				h.EntryPoint = n
			}
		}
	}
	return nil
}

// Update changes the vector for an existing node and re-inserts it in the graph.
func (h *HNSWIndex) Update(id int, vector []float32) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	node, exists := h.Nodes[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	if len(vector) != h.Dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(vector), h.Dimension)
	}
	// Normalize vector if using cosine distance.
	if h.DistanceName == "cosine" {
		core.NormalizeVector(vector)
	}
	h.removeNodeLinks(node)
	node.Vector = vector
	node.Links = make(map[int][]*Node)
	node.ReverseLinks = make(map[int][]*Node)
	h.insertNode(node, h.Ef)
	return nil
}

// BulkAdd inserts multiple vectors into the index at once.
func (h *HNSWIndex) BulkAdd(vectors map[int][]float32) error {
	// Normalize all vectors in batch if using cosine similarity.
	if h.DistanceName == "cosine" {
		var vecs [][]float32
		for _, vector := range vectors {
			if len(vector) != h.Dimension {
				return fmt.Errorf("vector dimension %d does not match index dimension %d",
					len(vector), h.Dimension)
			}
			vecs = append(vecs, vector)
		}
		core.NormalizeBatch(vecs)
	}

	nodesSlice := make([]*Node, 0, len(vectors))
	for id, vector := range vectors {
		if len(vector) != h.Dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d",
				len(vector), h.Dimension, id)
		}
		if _, exists := h.Nodes[id]; exists {
			return fmt.Errorf("id %d already exists", id)
		}
		level := h.randomLevel()
		newNode := &Node{
			ID:           id,
			Vector:       vector,
			Level:        level,
			Links:        make(map[int][]*Node),
			ReverseLinks: make(map[int][]*Node),
		}
		nodesSlice = append(nodesSlice, newNode)
	}
	// Sort nodes by level descending.
	sort.Slice(nodesSlice, func(i, j int) bool {
		return nodesSlice[i].Level > nodesSlice[j].Level
	})
	bulkEf := h.Ef
	h.Mu.Lock()
	defer h.Mu.Unlock()
	for _, newNode := range nodesSlice {
		h.Nodes[newNode.ID] = newNode
		if h.EntryPoint == nil {
			h.EntryPoint = newNode
			h.MaxLevel = newNode.Level
		} else {
			if newNode.Level > h.MaxLevel {
				h.EntryPoint = newNode
				h.MaxLevel = newNode.Level
			}
			h.insertNode(newNode, bulkEf)
		}
	}
	return nil
}

// BulkDelete removes multiple nodes from the index.
func (h *HNSWIndex) BulkDelete(ids []int) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	for _, id := range ids {
		node, exists := h.Nodes[id]
		if !exists {
			continue
		}
		h.removeNodeLinks(node)
		delete(h.Nodes, id)
	}
	// Clean up links in remaining nodes.
	for _, n := range h.Nodes {
		for L, neighbors := range n.Links {
			newNeighbors := make([]*Node, 0, len(neighbors))
			for _, neighbor := range neighbors {
				if _, exists := h.Nodes[neighbor.ID]; exists {
					newNeighbors = append(newNeighbors, neighbor)
				}
			}
			n.Links[L] = newNeighbors
		}
	}
	// Update the entry point.
	h.EntryPoint = nil
	for _, n := range h.Nodes {
		if h.EntryPoint == nil || n.Level > h.EntryPoint.Level {
			h.EntryPoint = n
		}
	}
	return nil
}

// BulkUpdate updates multiple nodes with new vectors.
func (h *HNSWIndex) BulkUpdate(updates map[int][]float32) error {
	// Normalize vectors in batch if needed.
	if h.DistanceName == "cosine" {
		var vecs [][]float32
		for _, vector := range updates {
			if len(vector) != h.Dimension {
				return fmt.Errorf("vector dimension %d does not match index dimension %d",
					len(vector), h.Dimension)
			}
			vecs = append(vecs, vector)
		}
		core.NormalizeBatch(vecs)
	}

	h.Mu.Lock()
	defer h.Mu.Unlock()
	for id, vector := range updates {
		node, exists := h.Nodes[id]
		if !exists {
			continue
		}
		if len(vector) != h.Dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d",
				len(vector), h.Dimension, id)
		}
		h.removeNodeLinks(node)
		node.Vector = vector
		node.Links = make(map[int][]*Node)
		node.ReverseLinks = make(map[int][]*Node)
	}
	// Reinsert all nodes to rebuild links.
	allNodes := make([]*Node, 0, len(h.Nodes))
	for _, node := range h.Nodes {
		allNodes = append(allNodes, node)
	}
	sort.Slice(allNodes, func(i, j int) bool {
		return allNodes[i].Level > allNodes[j].Level
	})
	h.EntryPoint = nil
	h.MaxLevel = -1
	for _, node := range allNodes {
		if h.EntryPoint == nil || node.Level > h.MaxLevel {
			h.EntryPoint = node
			h.MaxLevel = node.Level
		}
		h.insertNode(node, h.Ef)
	}
	return nil
}

// Search finds the k-nearest neighbors of a given query vector.
func (h *HNSWIndex) Search(query []float32, k int) ([]core.Neighbor, error) {
	h.Mu.RLock()
	defer h.Mu.RUnlock()
	if len(query) != h.Dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d",
			len(query), h.Dimension)
	}
	if h.EntryPoint == nil {
		return nil, errors.New("index is empty")
	}

	// Copy query to avoid modifying the original vector.
	queryCopy := make([]float32, len(query))
	copy(queryCopy, query)
	if h.DistanceName == "cosine" {
		core.NormalizeVector(queryCopy)
	}
	query = queryCopy

	// Greedy search down from the top layer.
	current := h.EntryPoint
	for L := h.MaxLevel; L > 0; L-- {
		changed := true
		for changed {
			changed = false
			for _, neighbor := range current.Links[L] {
				if h.Distance(query, neighbor.Vector) < h.Distance(query, current.Vector) {
					current = neighbor
					changed = true
				}
			}
		}
	}
	// Search in the base layer (level 0) for candidates.
	candidates := h.searchLayer(query, current, 0, h.Ef, h.Distance)
	if len(candidates) < k {
		// Use fallback to gather more candidates if needed.
		candidateIDs := make(map[int]bool)
		for _, c := range candidates {
			candidateIDs[c.node.ID] = true
		}

		fallbackSize := k - len(candidates)
		var keys []int
		for id := range h.Nodes {
			keys = append(keys, id)
		}
		sort.Ints(keys)
		nodesSlice := make([]*Node, 0, len(h.Nodes))
		for _, id := range keys {
			node := h.Nodes[id]
			if candidateIDs[node.ID] {
				continue
			}
			nodesSlice = append(nodesSlice, node)
		}

		numWorkers := runtime.NumCPU()
		if numWorkers > len(nodesSlice) {
			numWorkers = len(nodesSlice)
		}
		chunkSize := (len(nodesSlice) + numWorkers - 1) / numWorkers
		resultsCh := make(chan candidateMaxHeap, numWorkers)
		var wg sync.WaitGroup

		// Run parallel fallback search.
		for i := 0; i < numWorkers; i++ {
			start := i * chunkSize
			end := start + chunkSize
			if end > len(nodesSlice) {
				end = len(nodesSlice)
			}
			wg.Add(1)
			go func(nodesChunk []*Node) {
				defer wg.Done()
				localHeap := candidateMaxHeap{}
				heap.Init(&localHeap)
				for _, node := range nodesChunk {
					d := h.Distance(query, node.Vector)
					cand := candidate{node, d}
					if localHeap.Len() < fallbackSize {
						heap.Push(&localHeap, cand)
					} else if localHeap.Len() > 0 && d < localHeap[0].dist {
						heap.Pop(&localHeap)
						heap.Push(&localHeap, cand)
					}
				}
				resultsCh <- localHeap
			}(nodesSlice[start:end])
		}
		wg.Wait()
		close(resultsCh)

		finalHeap := candidateMaxHeap{}
		heap.Init(&finalHeap)
		// Merge results from all workers.
		for partialHeap := range resultsCh {
			for partialHeap.Len() > 0 {
				cand := heap.Pop(&partialHeap).(candidate)
				if finalHeap.Len() < fallbackSize {
					heap.Push(&finalHeap, cand)
				} else if finalHeap.Len() > 0 && cand.dist < finalHeap[0].dist {
					heap.Pop(&finalHeap)
					heap.Push(&finalHeap, cand)
				}
			}
		}
		fallbackCandidates := make([]candidate, finalHeap.Len())
		for i := range fallbackCandidates {
			fallbackCandidates[i] = heap.Pop(&finalHeap).(candidate)
		}
		candidates = append(candidates, fallbackCandidates...)
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].dist == candidates[j].dist {
				return candidates[i].node.ID < candidates[j].node.ID
			}
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
	return results, nil
}

// Stats returns simple statistics about the index.
func (h *HNSWIndex) Stats() core.IndexStats {
	h.Mu.RLock()
	defer h.Mu.RUnlock()
	count := len(h.Nodes)
	stats := core.IndexStats{
		Count:     count,
		Dimension: h.Dimension,
		Distance:  h.DistanceName,
	}
	return stats
}

// Save writes the index to disk using gob encoding.
func (h *HNSWIndex) Save(path string) error {
	h.Mu.RLock()
	defer h.Mu.RUnlock()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if err := enc.Encode(h); err != nil {
		return err
	}
	log.Info().Msgf("Index saved to %s", path)
	return nil
}

// Load reads the index from disk using gob decoding.
func (h *HNSWIndex) Load(path string) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	if err := dec.Decode(h); err != nil {
		return err
	}
	log.Info().Msgf("Index loaded from %s", path)
	return nil
}

// Check interface compliance at compile time.
var _ core.Index = (*HNSWIndex)(nil)

// init registers types for gob encoding.
func init() {
	gob.Register(serializedIndex{})
	gob.Register(serializedNode{})
	gob.Register(&HNSWIndex{})
	gob.Register(&Node{})
	log.Debug().Msg("Registered HNSWIndex and Node types for Gob encoding")
}

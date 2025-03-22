package hnsw

import (
	"bytes"
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

// candidateMinHeap implements a min–heap.
type candidateMinHeap []candidate

func (h candidateMinHeap) Len() int            { return len(h) }
func (h candidateMinHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h candidateMinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *candidateMinHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *candidateMinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// candidateMaxHeap implements a max–heap.
type candidateMaxHeap []candidate

func (h candidateMaxHeap) Len() int            { return len(h) }
func (h candidateMaxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h candidateMaxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *candidateMaxHeap) Push(x interface{}) { *h = append(*h, x.(candidate)) }
func (h *candidateMaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// Node represents an element in the HNSW graph.
type Node struct {
	ID           int
	Vector       []float32
	Level        int
	Links        map[int][]*Node // exported for serialization and rebuild
	ReverseLinks map[int][]*Node // exported for rebuild
}

// HNSWIndex implements a simplified HNSW index.
type HNSWIndex struct {
	// Mu is used for concurrency control and is skipped during serialization.
	Mu sync.RWMutex `gob:"-"`

	Dimension  int
	EntryPoint *Node
	MaxLevel   int
	Nodes      map[int]*Node

	M  int // maximum number of neighbors per node
	Ef int // search parameter

	// Distance is the function used internally for computing distances.
	Distance core.DistanceFunc
	// DistanceName is a human–readable name for the distance metric.
	DistanceName string
}

func init() {
	rand.New(rand.NewSource(time.Now().UnixNano()))
	log.Debug().Msg("Initialized random seed for HNSW index")
}

// NewHNSW creates a new HNSW index with the given parameters.
func NewHNSW(dimension int, M int, ef int, distance core.DistanceFunc, distanceName string) *HNSWIndex {
	log.Info().Msgf("Creating new HNSW index with dimension=%d, M=%d, ef=%d, distance=%s", dimension, M, ef, distanceName)
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

// randomLevel generates a random level for a new node using an exponential distribution.
func (h *HNSWIndex) randomLevel() int {
	if h.M <= 1 {
		return 0
	}
	level := int(-math.Log(rand.Float64()) / math.Log(float64(h.M)))
	log.Debug().Msgf("Generated random level %d", level)
	return level
}

// --- Custom Gob Serialization with Flattened Representation ---

type serializedNode struct {
	ID     int
	Vector []float32
	Level  int
	Links  map[int][]int // store neighbor IDs per level
}

type serializedIndex struct {
	Dimension    int
	M            int
	Ef           int
	Nodes        map[int]serializedNode
	EntryPoint   int // ID of entry point; 0 if nil
	MaxLevel     int
	DistanceName string
}

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
		return nil, err
	}
	return buf.Bytes(), nil
}

func (h *HNSWIndex) GobDecode(data []byte) error {
	var si serializedIndex
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&si); err != nil {
		return err
	}
	h.Dimension = si.Dimension
	h.M = si.M
	h.Ef = si.Ef
	h.MaxLevel = si.MaxLevel
	h.DistanceName = si.DistanceName
	h.Nodes = make(map[int]*Node)
	for id, sn := range si.Nodes {
		h.Nodes[id] = &Node{
			ID:           sn.ID,
			Vector:       sn.Vector,
			Level:        sn.Level,
			Links:        make(map[int][]*Node),
			ReverseLinks: make(map[int][]*Node),
		}
	}
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

// --- Core Methods ---

func selectM(candidates []candidate, M int) []candidate {
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})
	if len(candidates) > M {
		return candidates[:M]
	}
	return candidates
}

func selectNodes(nodes []*Node, vec []float32, M int, distance func([]float32, []float32) float64) []*Node {
	sort.Slice(nodes, func(i, j int) bool {
		return distance(vec, nodes[i].Vector) < distance(vec, nodes[j].Vector)
	})
	if len(nodes) > M {
		return nodes[:M]
	}
	return nodes
}

func removeFromSlice(slice []*Node, target *Node) []*Node {
	newSlice := slice[:0]
	for _, n := range slice {
		if n != target {
			newSlice = append(newSlice, n)
		}
	}
	return newSlice
}

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

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (h *HNSWIndex) insertNode(n *Node, searchEf int) {
	if h.EntryPoint == nil {
		h.EntryPoint = n
		h.MaxLevel = n.Level
		return
	}
	if n.Level > h.MaxLevel {
		h.EntryPoint = n
		h.MaxLevel = n.Level
	}
	current := h.EntryPoint
	// Greedy descent on levels above n.Level.
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
	// For levels from min(n.Level, h.MaxLevel) down to 0.
	for L := minInt(n.Level, h.MaxLevel); L >= 0; L-- {
		candList := h.searchLayer(n.Vector, current, L, searchEf, h.Distance)
		selectedCands := selectM(candList, h.M)
		selectedNodes := make([]*Node, len(selectedCands))
		for i, cand := range selectedCands {
			selectedNodes[i] = cand.node
		}
		n.Links[L] = selectedNodes
		for _, neighbor := range selectedNodes {
			neighbor.Links[L] = append(neighbor.Links[L], n)
			neighbor.ReverseLinks[L] = append(neighbor.ReverseLinks[L], n)
			if len(neighbor.Links[L]) > h.M {
				neighbor.Links[L] = selectNodes(neighbor.Links[L], neighbor.Vector, h.M, h.Distance)
			}
		}
		if len(candList) > 0 {
			current = candList[0].node
		}
	}
}

func (h *HNSWIndex) searchLayer(query []float32, entrypoint *Node, level int, ef int, distance func([]float32, []float32) float64) []candidate {
	visited := map[int]bool{entrypoint.ID: true}
	d0 := distance(query, entrypoint.Vector)
	candQueue := candidateMinHeap{{entrypoint, d0}}
	heap.Init(&candQueue)
	resultQueue := candidateMaxHeap{{entrypoint, d0}}
	heap.Init(&resultQueue)
	for candQueue.Len() > 0 {
		current := candQueue[0]
		worstResult := resultQueue[0]
		if current.dist > worstResult.dist {
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
	results := make([]candidate, resultQueue.Len())
	for i := range results {
		results[i] = heap.Pop(&resultQueue).(candidate)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})
	return results
}

func (h *HNSWIndex) Add(id int, vector []float32) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	if len(vector) != h.Dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), h.Dimension)
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

func (h *HNSWIndex) Delete(id int) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	node, exists := h.Nodes[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	h.removeNodeLinks(node)
	delete(h.Nodes, id)
	if h.EntryPoint != nil && h.EntryPoint.ID == id {
		h.EntryPoint = nil
		for _, n := range h.Nodes {
			h.EntryPoint = n
			break
		}
	}
	return nil
}

func (h *HNSWIndex) Update(id int, vector []float32) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	node, exists := h.Nodes[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	if len(vector) != h.Dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), h.Dimension)
	}
	h.removeNodeLinks(node)
	node.Vector = vector
	node.Level = h.randomLevel()
	node.Links = make(map[int][]*Node)
	node.ReverseLinks = make(map[int][]*Node)
	h.insertNode(node, h.Ef)
	return nil
}

func (h *HNSWIndex) BulkAdd(vectors map[int][]float32) error {
	nodesSlice := make([]*Node, 0, len(vectors))
	for id, vector := range vectors {
		if len(vector) != h.Dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), h.Dimension, id)
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
	sort.Slice(nodesSlice, func(i, j int) bool {
		return nodesSlice[i].Level > nodesSlice[j].Level
	})
	bulkEf := h.Ef
	if bulkEf > 16 {
		bulkEf = 16
	}
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

func (h *HNSWIndex) BulkDelete(ids []int) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	for _, id := range ids {
		if _, exists := h.Nodes[id]; !exists {
			continue
		}
		for _, n := range h.Nodes {
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
		delete(h.Nodes, id)
		if h.EntryPoint != nil && h.EntryPoint.ID == id {
			h.EntryPoint = nil
			for _, n := range h.Nodes {
				h.EntryPoint = n
				break
			}
		}
	}
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
	return nil
}

func (h *HNSWIndex) BulkUpdate(updates map[int][]float32) error {
	h.Mu.Lock()
	defer h.Mu.Unlock()
	for id, vector := range updates {
		node, exists := h.Nodes[id]
		if !exists {
			continue
		}
		if len(vector) != h.Dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d", len(vector), h.Dimension, id)
		}
		h.removeNodeLinks(node)
		node.Vector = vector
		node.Level = h.randomLevel()
		node.Links = make(map[int][]*Node)
		node.ReverseLinks = make(map[int][]*Node)
	}
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
		h.insertNode(node, h.Ef)
	}
	return nil
}

// Search returns the k nearest neighbors using the internally set distance.
func (h *HNSWIndex) Search(query []float32, k int) ([]core.Neighbor, error) {
	h.Mu.RLock()
	defer h.Mu.RUnlock()
	if len(query) != h.Dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), h.Dimension)
	}
	if h.EntryPoint == nil {
		return nil, errors.New("index is empty")
	}
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
	candidates := h.searchLayer(query, current, 0, h.Ef, h.Distance)
	if len(candidates) < k {
		candidateIDs := make(map[int]bool)
		for _, c := range candidates {
			candidateIDs[c.node.ID] = true
		}
		fallbackHeap := candidateMaxHeap{}
		heap.Init(&fallbackHeap)
		for _, node := range h.Nodes {
			if candidateIDs[node.ID] {
				continue
			}
			d := h.Distance(query, node.Vector)
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
	return results, nil
}

func (h *HNSWIndex) Stats() core.IndexStats {
	h.Mu.RLock()
	defer h.Mu.RUnlock()
	count := len(h.Nodes)
	size := count * h.Dimension * 4
	stats := core.IndexStats{
		Count:     count,
		Dimension: h.Dimension,
		Size:      size,
		Distance:  h.DistanceName,
	}
	return stats
}

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

var _ core.Index = (*HNSWIndex)(nil)

func init() {
	gob.Register(serializedIndex{})
	gob.Register(serializedNode{})
	gob.Register(&HNSWIndex{})
	gob.Register(&Node{})
	log.Debug().Msg("Registered HNSWIndex and Node types for Gob encoding")
}

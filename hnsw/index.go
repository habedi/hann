// Package hnsw implements the HNSW graph index.
package hnsw

import (
	"bytes"
	"container/heap"
	"encoding/gob"
	"errors"
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

// seededRand is a global random number generator used for level generation.
var seededRand = rand.New(rand.NewSource(core.GetSeed()))
var seededRandMu sync.Mutex

// maxLevelCap is the upper bound for a node's level.
const maxLevelCap = 32

// Default construction parameters used by New when no option overrides them.
const (
	defaultM              = 16
	defaultEf             = 100
	defaultEfConstruction = 200
)

// candidate represents a potential neighbor with its distance.
type candidate struct {
	node *node   // reference to the candidate node
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

// node represents a vector in the HNSW graph along with its links.
type node struct {
	mu           sync.Mutex      // guards Links and ReverseLinks during a parallel BulkAdd
	ID           int             // unique identifier of the node
	Vector       []float32       // vector data
	Level        int             // node level in the hierarchy
	Links        map[int][]*node // links to neighbors at each level
	ReverseLinks map[int][]*node // reverse links from neighbors
}

// bulkState carries the shared synchronization state for the parallel phase
// of BulkAdd. It exists only while BulkAdd holds the index write lock, so
// external readers never run concurrently with it. A nil *bulkState selects
// the sequential path everywhere, which behaves exactly like the
// single-item operations.
type bulkState struct {
	epMu sync.Mutex // guards entryPoint and maxLevel between inserters
}

// containsNode reports whether the list holds the target.
func containsNode(list []*node, target *node) bool {
	for _, n := range list {
		if n == target {
			return true
		}
	}
	return false
}

// neighborList returns a node's neighbors at a level. The sequential path
// returns the live slice. The parallel path returns a copy taken under the
// node's lock, because a concurrent trim can rewrite the live slice in
// place.
func (h *Index) neighborList(n *node, level int, par *bulkState) []*node {
	if par == nil {
		return n.Links[level]
	}
	n.mu.Lock()
	links := append([]*node(nil), n.Links[level]...)
	n.mu.Unlock()
	return links
}

// linkLen returns the number of neighbors a node has at a level.
func (h *Index) linkLen(n *node, level int, par *bulkState) int {
	if par == nil {
		return len(n.Links[level])
	}
	n.mu.Lock()
	length := len(n.Links[level])
	n.mu.Unlock()
	return length
}

// addEdge records the directed edge from one node to another, on both sides
// of the bookkeeping. The parallel path takes one node lock at a time, so
// no inserter ever holds two locks, and it skips an edge that a concurrent
// inserter already created. A trim that interleaves between the two halves
// can leave a reverse record without a matching link. That is tolerated:
// every reader of reverse links treats a record without a link as a no-op.
func (h *Index) addEdge(from, to *node, level int, par *bulkState) {
	if par == nil {
		from.Links[level] = append(from.Links[level], to)
		to.ReverseLinks[level] = append(to.ReverseLinks[level], from)
		return
	}
	from.mu.Lock()
	if containsNode(from.Links[level], to) {
		from.mu.Unlock()
		return
	}
	from.Links[level] = append(from.Links[level], to)
	from.mu.Unlock()
	to.mu.Lock()
	to.ReverseLinks[level] = append(to.ReverseLinks[level], from)
	to.mu.Unlock()
}

// Index is the HNSW graph index. Create it with New. The zero value is
// usable only as a target for Load.
type Index struct {
	mu               sync.RWMutex  // mutex to control concurrent access
	fallbackSearches atomic.Int64  // searches that fell back to a brute-force scan
	dimension        int           // dimension of the vectors
	entryPoint       *node         // starting point for searches
	maxLevel         int           // current maximum level in the graph
	nodes            map[int]*node // map of node id to node pointer
	nodesByLevel     map[int]map[int]struct{}
	m                int         // maximum number of neighbors per node
	ef               int         // search parameter controlling search depth
	efConstruction   int         // search depth used while building the graph
	metric           core.Metric // distance metric used by the index
	exhaustiveSearch bool        // flag for performing exhaustive search during searchLayer
}

// Option configures an Index during construction with New.
type Option func(*Index)

// WithM sets the maximum number of neighbors per node. The default is 16.
func WithM(m int) Option {
	return func(h *Index) { h.m = m }
}

// WithEf sets the search depth parameter. The default is 100.
func WithEf(ef int) Option {
	return func(h *Index) { h.ef = ef }
}

// WithEfConstruction sets the search depth used while building the graph.
// A larger value gives each new node a wider pool of candidate neighbors.
// That improves graph quality but makes insertion slower. The default is
// 200.
func WithEfConstruction(efConstruction int) Option {
	return func(h *Index) { h.efConstruction = efConstruction }
}

// WithMetric sets the distance metric. The default is core.Euclidean.
func WithMetric(metric core.Metric) Option {
	return func(h *Index) { h.metric = metric }
}

// WithExhaustiveSearch turns exhaustive layer search on or off. It is off
// by default. Turning it on makes layer exploration exact but slower.
func WithExhaustiveSearch(on bool) Option {
	return func(h *Index) { h.exhaustiveSearch = on }
}

// New creates an HNSW index for vectors of the given dimension. It applies
// the given options over the defaults: M 16, Ef 100, EfConstruction 200,
// the Euclidean metric, and exhaustive search off. It returns an error when
// the dimension is not positive, M is below 2, Ef is below 1,
// EfConstruction is below 1, or the metric is the zero value.
func New(dimension int, opts ...Option) (*Index, error) {
	h := &Index{
		dimension:      dimension,
		nodes:          make(map[int]*node),
		nodesByLevel:   make(map[int]map[int]struct{}),
		maxLevel:       -1,
		m:              defaultM,
		ef:             defaultEf,
		efConstruction: defaultEfConstruction,
		metric:         core.Euclidean,
	}
	for _, opt := range opts {
		opt(h)
	}
	if dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", dimension)
	}
	if h.m < 2 {
		return nil, fmt.Errorf("parameter M must be at least 2, got %d", h.m)
	}
	if h.ef < 1 {
		return nil, fmt.Errorf("parameter Ef must be at least 1, got %d", h.ef)
	}
	if h.efConstruction < 1 {
		return nil, fmt.Errorf("parameter EfConstruction must be at least 1, got %d", h.efConstruction)
	}
	if h.metric.IsZero() {
		return nil, errors.New("metric must not be the zero value")
	}
	return h, nil
}

// SetEf changes the search breadth used by Search. Larger values improve
// recall but raise latency. The setting takes effect for searches that
// start after the call. It returns an error when ef is not positive.
func (h *Index) SetEf(ef int) error {
	if ef < 1 {
		return fmt.Errorf("parameter Ef must be at least 1, got %d", ef)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.ef = ef
	return nil
}

// randomLevel picks a random level for a new node from an exponential
// distribution.
func (h *Index) randomLevel() int {
	if h.m <= 1 {
		return 0
	}
	seededRandMu.Lock()
	r := seededRand.Float64()
	seededRandMu.Unlock()
	return levelFromUnit(r, h.m)
}

// levelFromUnit maps a uniform draw r in [0, 1) to a node level. A draw of
// exactly 0 has an infinite logarithm, and converting an infinite float to
// int is implementation-defined, so it maps straight to the cap.
func levelFromUnit(r float64, m int) int {
	if r <= 0 {
		return maxLevelCap
	}
	level := int(-math.Log(r) / math.Log(float64(m)))
	if level > maxLevelCap {
		level = maxLevelCap
	}
	return level
}

// serializedNode is used to store a node during gob encoding/decoding.
type serializedNode struct {
	ID     int           // node id
	Vector []float32     // vector data
	Level  int           // node level
	Links  map[int][]int // neighbor ids at each level
}

// serializedIndex is the serializable version of the Index.
type serializedIndex struct {
	Dimension        int                    // dimension of the index
	M                int                    // maximum neighbors per node
	Ef               int                    // search parameter
	Nodes            map[int]serializedNode // serialized nodes
	EntryPoint       int                    // id of the entry point node
	MaxLevel         int                    // maximum level in the graph
	DistanceName     string                 // name of the distance metric
	ExhaustiveSearch bool                   // exhaustive layer search flag
	HasEntryPoint    bool                   // whether EntryPoint holds a valid node id
	FormatVersion    int                    // on-disk format version
	EfConstruction   int                    // search depth used while building the graph
}

// formatVersion is the on-disk format version written by GobEncode. Files
// written before the field existed decode it as zero and are accepted.
// Files written by a newer version of the format are rejected on load.
const formatVersion = 1

// GobEncode serializes the Index using the gob encoder.
func (h *Index) GobEncode() ([]byte, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	si := serializedIndex{
		Dimension:        h.dimension,
		M:                h.m,
		Ef:               h.ef,
		Nodes:            make(map[int]serializedNode),
		EntryPoint:       0,
		MaxLevel:         h.maxLevel,
		DistanceName:     h.metric.Name(),
		ExhaustiveSearch: h.exhaustiveSearch,
		FormatVersion:    formatVersion,
		EfConstruction:   h.efConstruction,
	}
	for id, n := range h.nodes {
		sn := serializedNode{
			ID:     n.ID,
			Vector: n.Vector,
			Level:  n.Level,
			Links:  make(map[int][]int),
		}
		// Store neighbor ids for each level.
		for level, neighbors := range n.Links {
			for _, nb := range neighbors {
				sn.Links[level] = append(sn.Links[level], nb.ID)
			}
		}
		si.Nodes[id] = sn
	}
	if h.entryPoint != nil {
		si.EntryPoint = h.entryPoint.ID
		si.HasEntryPoint = true
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(si); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode deserializes data into the Index using the gob decoder.
func (h *Index) GobDecode(data []byte) error {
	var si serializedIndex
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&si); err != nil {
		return err
	}
	if si.FormatVersion > formatVersion {
		return fmt.Errorf("index file has format version %d, but this build supports up to %d",
			si.FormatVersion, formatVersion)
	}
	h.dimension = si.Dimension
	// A corrupt or crafted file can carry parameters New would reject, and
	// they silently break graph maintenance. Fall back to the defaults for
	// any out-of-range value.
	h.m = si.M
	if h.m < 2 {
		h.m = defaultM
	}
	h.ef = si.Ef
	if h.ef < 1 {
		h.ef = defaultEf
	}
	h.maxLevel = si.MaxLevel
	h.exhaustiveSearch = si.ExhaustiveSearch
	// Files written before the field existed decode it as zero. Fall back
	// to the default so later insertions search with a usable depth.
	h.efConstruction = si.EfConstruction
	if h.efConstruction < 1 {
		h.efConstruction = defaultEfConstruction
	}
	// Restore the metric from its name. An index that was built with a
	// custom metric keeps it when the name is unknown.
	if metric, ok := core.MetricByName(si.DistanceName); ok {
		h.metric = metric
	} else if h.metric.IsZero() {
		return fmt.Errorf("unknown metric %q in serialized index", si.DistanceName)
	}
	h.nodes = make(map[int]*node)
	// Recreate nodes from the serialized data.
	for id, sn := range si.Nodes {
		h.nodes[id] = &node{
			ID:           sn.ID,
			Vector:       sn.Vector,
			Level:        sn.Level,
			Links:        make(map[int][]*node),
			ReverseLinks: make(map[int][]*node),
		}
	}
	// Restore neighbor pointers.
	for id, sn := range si.Nodes {
		n := h.nodes[id]
		for level, nbIDs := range sn.Links {
			for _, nbID := range nbIDs {
				if nb, exists := h.nodes[nbID]; exists {
					n.Links[level] = append(n.Links[level], nb)
				}
			}
		}
	}
	// Rebuild reverse links.
	for _, n := range h.nodes {
		for level, neighbors := range n.Links {
			for _, nb := range neighbors {
				nb.ReverseLinks[level] = append(nb.ReverseLinks[level], n)
			}
		}
	}
	// Rebuild the per-level bookkeeping used by Delete to pick a new entry point.
	h.nodesByLevel = make(map[int]map[int]struct{})
	for id, n := range h.nodes {
		if _, ok := h.nodesByLevel[n.Level]; !ok {
			h.nodesByLevel[n.Level] = make(map[int]struct{})
		}
		h.nodesByLevel[n.Level][id] = struct{}{}
	}
	if si.HasEntryPoint {
		h.entryPoint = h.nodes[si.EntryPoint]
	} else if si.EntryPoint != 0 {
		// Legacy files carry no flag and use id 0 as the nil sentinel.
		h.entryPoint = h.nodes[si.EntryPoint]
	} else if n, ok := h.nodes[0]; ok {
		// A legacy file whose entry point was the node with id 0 stored the
		// sentinel value. Recover by using that node.
		h.entryPoint = n
	} else {
		h.entryPoint = nil
	}
	return nil
}

// selectNeighborsHeuristic picks up to M neighbors from the candidates. It
// uses the neighbor selection heuristic from the HNSW paper. Candidates are
// considered nearest first. A candidate is selected only when it is closer
// to the query than to every neighbor selected before it. A candidate that
// fails the check is set aside. This keeps some links pointing across
// sparse regions instead of piling every link into the densest cluster.
// That is what keeps the level 0 graph connected on clustered data. Any
// remaining slots are then filled with the nearest of the set-aside
// candidates. The candidate distances and the given distance function must
// both be in rank space.
func selectNeighborsHeuristic(candidates []candidate, M int, distance core.DistanceFunc) ([]candidate, error) {
	sorted := make([]candidate, len(candidates))
	copy(sorted, candidates)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].dist == sorted[j].dist {
			return sorted[i].node.ID < sorted[j].node.ID
		}
		return sorted[i].dist < sorted[j].dist
	})
	result := make([]candidate, 0, minInt(len(sorted), M))
	var discarded []candidate
	for _, e := range sorted {
		if len(result) == M {
			break
		}
		closerToQuery := true
		for _, r := range result {
			d, err := distance(e.node.Vector, r.node.Vector)
			if err != nil {
				return nil, err
			}
			if d <= e.dist {
				closerToQuery = false
				break
			}
		}
		if closerToQuery {
			result = append(result, e)
		} else {
			discarded = append(discarded, e)
		}
	}
	// Keep pruned connections: fill the remaining slots with the nearest of
	// the discarded candidates. They are already in nearest-first order.
	for _, e := range discarded {
		if len(result) == M {
			break
		}
		result = append(result, e)
	}
	return result, nil
}

// removeFromSlice removes a target node from a slice of nodes.
func removeFromSlice(slice []*node, target *node) []*node {
	newSlice := slice[:0]
	for _, n := range slice {
		if n != target {
			newSlice = append(newSlice, n)
		}
	}
	return newSlice
}

// difference returns nodes in a that are not in b.
func difference(a, b []*node) []*node {
	set := make(map[int]bool)
	for _, n := range b {
		set[n.ID] = true
	}
	var diff []*node
	for _, n := range a {
		if !set[n.ID] {
			diff = append(diff, n)
		}
	}
	return diff
}

// trimNeighborLinks cuts a node's neighbors at a level down to at most M.
// The kept neighbors are chosen by the neighbor selection heuristic over
// the current links. The parallel path holds the node's lock for the whole
// selection, so the choice is made against a consistent list, and it takes
// the removed nodes' locks one at a time afterward, so no two locks are
// ever held together.
func trimNeighborLinks(n *node, level, M int, distance core.DistanceFunc, par *bulkState) error {
	if par != nil {
		n.mu.Lock()
	}
	original := n.Links[level]
	cands := make([]candidate, len(original))
	for i, nb := range original {
		d, err := distance(n.Vector, nb.Vector)
		if err != nil {
			if par != nil {
				n.mu.Unlock()
			}
			return err
		}
		cands[i] = candidate{nb, d}
	}
	selected, err := selectNeighborsHeuristic(cands, M, distance)
	if err != nil {
		if par != nil {
			n.mu.Unlock()
		}
		return err
	}
	trimmed := make([]*node, len(selected))
	for i, c := range selected {
		trimmed[i] = c.node
	}
	removed := difference(original, trimmed)
	n.Links[level] = trimmed
	if par != nil {
		n.mu.Unlock()
	}
	for _, r := range removed {
		if par != nil {
			r.mu.Lock()
		}
		r.ReverseLinks[level] = removeFromSlice(r.ReverseLinks[level], n)
		if par != nil {
			r.mu.Unlock()
		}
	}
	return nil
}

// removeNodeLinks removes all links of a node from the graph.
func (h *Index) removeNodeLinks(n *node) {
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

// resetEntryPoint recomputes maxLevel and picks a new entry point from the
// per-level bookkeeping. It skips the node with the given id. It leaves the
// entry point nil when no other node exists. The caller must hold the lock.
func (h *Index) resetEntryPoint(excludeID int) {
	h.entryPoint = nil
	h.maxLevel = -1
	for level, ids := range h.nodesByLevel {
		if level <= h.maxLevel {
			continue
		}
		for id := range ids {
			if id == excludeID {
				continue
			}
			h.maxLevel = level
			h.entryPoint = h.nodes[id]
			break
		}
	}
}

// minInt returns the smaller of two integers.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// rollbackInsert removes a node whose insertNode call failed partway. The
// partial link state is symmetric at every error point, so removeNodeLinks
// unlinks the node completely. The entry point is restored when the failed
// node had claimed it. The caller must hold the lock, and the node must be
// present in the maps.
func (h *Index) rollbackInsert(n *node) {
	h.removeNodeLinks(n)
	delete(h.nodes, n.ID)
	if level, ok := h.nodesByLevel[n.Level]; ok {
		delete(level, n.ID)
		if len(level) == 0 {
			delete(h.nodesByLevel, n.Level)
		}
	}
	if h.entryPoint == n {
		h.resetEntryPoint(n.ID)
	}
}

// insertNode adds a node into the HNSW graph, updating links as needed.
func (h *Index) insertNode(n *node, searchEf int, par *bulkState) error {
	if par != nil {
		par.epMu.Lock()
	}
	// If index is empty, set this node as entry point.
	if h.entryPoint == nil {
		h.entryPoint = n
		h.maxLevel = n.Level
		if par != nil {
			par.epMu.Unlock()
		}
		return nil
	}
	// Seed the search from the current entry point before it may change.
	// This way the node being inserted never becomes its own search seed.
	current := h.entryPoint
	// Update entry point if the new node has a higher level.
	if n.Level > h.maxLevel {
		h.entryPoint = n
		h.maxLevel = n.Level
	}
	top := h.maxLevel
	if par != nil {
		par.epMu.Unlock()
	}
	// Navigate the graph from the top level down to the node's level.
	for L := top; L > n.Level; L-- {
		changed := true
		for changed {
			changed = false
			for _, neighbor := range h.neighborList(current, L, par) {
				distNeighbor, err := h.metric.Rank(n.Vector, neighbor.Vector)
				if err != nil {
					return err
				}
				distCurrent, err := h.metric.Rank(n.Vector, current.Vector)
				if err != nil {
					return err
				}
				if distNeighbor < distCurrent {
					current = neighbor
					changed = true
				}
			}
		}
	}
	// For each level where the new node will be inserted.
	for L := minInt(n.Level, top); L >= 0; L-- {
		candList, err := h.searchLayer(n.Vector, current, L, searchEf, h.metric.Rank, par)
		if err != nil {
			return err
		}
		selectedCands, err := selectNeighborsHeuristic(candList, h.m, h.metric.Rank)
		if err != nil {
			return err
		}
		// Wire both directions of every selected edge. The backlink records
		// let removeNodeLinks later remove n from the neighbor's links.
		for _, cand := range selectedCands {
			neighbor := cand.node
			h.addEdge(n, neighbor, L, par)
			h.addEdge(neighbor, n, L, par)
			if h.linkLen(neighbor, L, par) > h.m {
				if err := trimNeighborLinks(neighbor, L, h.m, h.metric.Rank, par); err != nil {
					return err
				}
			}
		}
		// The new node's own list needs the same cap check: concurrent
		// inserters append backlinks to it while its selected edges are
		// wired, so the unchecked appends above can push it past M. With no
		// concurrency the selected edges alone never exceed M.
		if h.linkLen(n, L, par) > h.m {
			if err := trimNeighborLinks(n, L, h.m, h.metric.Rank, par); err != nil {
				return err
			}
		}
		// Move the current pointer for the next level.
		if len(candList) > 0 {
			current = candList[0].node
		}
	}
	return nil
}

// searchLayer performs a search in the graph at a given level.
func (h *Index) searchLayer(query []float32, entrypoint *node, level int, ef int, distance core.DistanceFunc, par *bulkState) ([]candidate, error) {
	visited := map[int]bool{entrypoint.ID: true}
	d0, err := distance(query, entrypoint.Vector)
	if err != nil {
		return nil, err
	}
	candQueue := candidateMinHeap{{entrypoint, d0}}
	heap.Init(&candQueue)
	resultQueue := candidateMaxHeap{{entrypoint, d0}}
	heap.Init(&resultQueue)
	// Explore candidates while there are promising ones.
	for candQueue.Len() > 0 {
		current := candQueue[0]
		worstResult := resultQueue[0]
		if current.dist > worstResult.dist && !h.exhaustiveSearch {
			break
		}
		heap.Pop(&candQueue)
		for _, neighbor := range h.neighborList(current.node, level, par) {
			if visited[neighbor.ID] {
				continue
			}
			visited[neighbor.ID] = true
			d, err := distance(query, neighbor.Vector)
			if err != nil {
				return nil, err
			}
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
	return results, nil
}

// Add inserts a new vector into the index with a unique id.
func (h *Index) Add(id int, vector []float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(vector) != h.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(vector), h.dimension)
	}
	// Normalize when the metric requires normalized vectors.
	if h.metric.Normalizes() {
		core.NormalizeVector(vector)
	}
	if _, exists := h.nodes[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}
	level := h.randomLevel()
	newNode := &node{
		ID:           id,
		Vector:       vector,
		Level:        level,
		Links:        make(map[int][]*node),
		ReverseLinks: make(map[int][]*node),
	}
	h.nodes[id] = newNode
	if _, ok := h.nodesByLevel[level]; !ok {
		h.nodesByLevel[level] = make(map[int]struct{})
	}
	h.nodesByLevel[level][id] = struct{}{}
	if err := h.insertNode(newNode, h.efConstruction, nil); err != nil {
		h.rollbackInsert(newNode)
		return err
	}
	return nil
}

// Delete removes a vector from the index by its id.
func (h *Index) Delete(id int) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	n, exists := h.nodes[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	h.removeNodeLinks(n)
	delete(h.nodes, id)
	level := n.Level
	if _, ok := h.nodesByLevel[level]; ok {
		delete(h.nodesByLevel[level], id)
		if len(h.nodesByLevel[level]) == 0 {
			delete(h.nodesByLevel, level)
		}
	}
	// Update the entry point if necessary.
	if h.entryPoint != nil && h.entryPoint.ID == id {
		h.entryPoint = nil
		// Find the new max level without a full scan of the nodes.
		newMaxLevel := -1
		for l := range h.nodesByLevel {
			if l > newMaxLevel {
				newMaxLevel = l
			}
		}
		h.maxLevel = newMaxLevel

		if h.maxLevel != -1 {
			// Pick any node from the highest level.
			for newEntryPointID := range h.nodesByLevel[h.maxLevel] {
				h.entryPoint = h.nodes[newEntryPointID]
				break
			}
		}
	}
	return nil
}

// reinsertLocked replaces the vector of an existing node and relinks it in
// the graph. When the relinking fails, the node's previous vector, its
// links, and the entry point are restored, so a failed update leaves every
// stored vector in place and every id reachable as before. A neighbor list
// that another node trimmed during the attempt can come out shorter, but it
// still holds valid neighbors. The caller must hold the lock.
func (h *Index) reinsertLocked(n *node, vector []float32) error {
	oldVector := n.Vector
	oldEntryPoint := h.entryPoint
	oldMaxLevel := h.maxLevel
	// removeNodeLinks empties the link slices in place, so the restore
	// needs copies of the current link lists.
	oldLinks := make(map[int][]*node, len(n.Links))
	for level, neighbors := range n.Links {
		oldLinks[level] = append([]*node(nil), neighbors...)
	}
	oldReverse := make(map[int][]*node, len(n.ReverseLinks))
	for level, neighbors := range n.ReverseLinks {
		oldReverse[level] = append([]*node(nil), neighbors...)
	}

	h.removeNodeLinks(n)
	n.Vector = vector
	n.Links = make(map[int][]*node)
	n.ReverseLinks = make(map[int][]*node)
	// Reinsertion must not start the search at the node itself. Move the
	// entry point to another surviving node first.
	if h.entryPoint == n {
		h.resetEntryPoint(n.ID)
	}
	if h.entryPoint == nil {
		// The node is the only one in the index.
		h.entryPoint = n
		h.maxLevel = n.Level
		return nil
	}
	if err := h.insertNode(n, h.efConstruction, nil); err != nil {
		// Unlink whatever the failed reinsertion wired, then put the old
		// vector and the old links back on both sides of every edge.
		h.removeNodeLinks(n)
		n.Vector = oldVector
		n.Links = oldLinks
		n.ReverseLinks = oldReverse
		for level, neighbors := range oldReverse {
			for _, nb := range neighbors {
				nb.Links[level] = append(nb.Links[level], n)
			}
		}
		for level, neighbors := range oldLinks {
			for _, nb := range neighbors {
				nb.ReverseLinks[level] = append(nb.ReverseLinks[level], n)
			}
		}
		h.entryPoint = oldEntryPoint
		h.maxLevel = oldMaxLevel
		return err
	}
	return nil
}

// Update changes the vector for an existing node and re-inserts it in the
// graph. A failed update restores the previous state; see reinsertLocked.
func (h *Index) Update(id int, vector []float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	n, exists := h.nodes[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	if len(vector) != h.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d",
			len(vector), h.dimension)
	}
	// Normalize when the metric requires normalized vectors.
	if h.metric.Normalizes() {
		core.NormalizeVector(vector)
	}
	return h.reinsertLocked(n, vector)
}

// BulkAdd inserts multiple vectors into the index at once. The nodes are
// inserted concurrently while the call holds the index write lock, so the
// graph layout varies between runs, like it already did through map
// iteration order. Search quality is unaffected.
func (h *Index) BulkAdd(vectors map[int][]float32) error {
	// Normalize vectors in batch when the metric requires it.
	if h.metric.Normalizes() {
		var vecs [][]float32
		for _, vector := range vectors {
			if len(vector) != h.dimension {
				return fmt.Errorf("vector dimension %d does not match index dimension %d",
					len(vector), h.dimension)
			}
			vecs = append(vecs, vector)
		}
		core.NormalizeBatch(vecs)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// The duplicate check reads h.nodes, so it must happen under the lock.
	nodesSlice := make([]*node, 0, len(vectors))
	for id, vector := range vectors {
		if len(vector) != h.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d",
				len(vector), h.dimension, id)
		}
		if _, exists := h.nodes[id]; exists {
			return fmt.Errorf("id %d already exists", id)
		}
		level := h.randomLevel()
		newNode := &node{
			ID:           id,
			Vector:       vector,
			Level:        level,
			Links:        make(map[int][]*node),
			ReverseLinks: make(map[int][]*node),
		}
		nodesSlice = append(nodesSlice, newNode)
	}
	// Sort nodes by level descending.
	sort.Slice(nodesSlice, func(i, j int) bool {
		return nodesSlice[i].Level > nodesSlice[j].Level
	})
	bulkEf := h.efConstruction

	// Register every node up front, so the parallel phase below never
	// writes the index maps. An unwired node is invisible to the graph
	// walks, because nothing links to it yet.
	for _, newNode := range nodesSlice {
		h.nodes[newNode.ID] = newNode
		if _, ok := h.nodesByLevel[newNode.Level]; !ok {
			h.nodesByLevel[newNode.Level] = make(map[int]struct{})
		}
		h.nodesByLevel[newNode.Level][newNode.ID] = struct{}{}
	}
	rest := nodesSlice
	if h.entryPoint == nil && len(rest) > 0 {
		// Seed an empty index with the highest-level node, so every
		// concurrent inserter has an entry point to descend from.
		h.entryPoint = rest[0]
		h.maxLevel = rest[0].Level
		rest = rest[1:]
	}
	if len(rest) == 0 {
		return nil
	}

	// Insert the nodes concurrently. The index write lock is held for the
	// whole call, so only these workers touch the graph, synchronized by
	// the per-node locks. The graph layout depends on how the insertions
	// interleave, like it already depended on map iteration order above.
	par := &bulkState{}
	numWorkers := runtime.NumCPU()
	if numWorkers > len(rest) {
		numWorkers = len(rest)
	}
	var (
		next     atomic.Int64
		stop     atomic.Bool
		errMu    sync.Mutex
		firstErr error
	)
	status := make([]atomic.Int32, len(rest)) // 0 not attempted, 1 inserted, 2 failed
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= len(rest) || stop.Load() {
					return
				}
				if err := h.insertNode(rest[i], bulkEf, par); err != nil {
					status[i].Store(2)
					errMu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					errMu.Unlock()
					stop.Store(true)
					return
				}
				status[i].Store(1)
			}
		}()
	}
	wg.Wait()
	if firstErr == nil {
		return nil
	}
	// The nodes never attempted leave the maps again first: they have no
	// links, and removing them before the rollbacks keeps resetEntryPoint
	// from picking an unwired node.
	for i, newNode := range rest {
		if status[i].Load() != 0 {
			continue
		}
		delete(h.nodes, newNode.ID)
		if level, ok := h.nodesByLevel[newNode.Level]; ok {
			delete(level, newNode.ID)
			if len(level) == 0 {
				delete(h.nodesByLevel, newNode.Level)
			}
		}
	}
	// A failing element rolls back like a failed Add. The elements inserted
	// before the failure stay, which matches applying Add to every element
	// in turn.
	for i, newNode := range rest {
		if status[i].Load() == 2 {
			h.rollbackInsert(newNode)
		}
	}
	return firstErr
}

// BulkDelete removes multiple nodes from the index.
func (h *Index) BulkDelete(ids []int) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	for _, id := range ids {
		n, exists := h.nodes[id]
		if !exists {
			continue
		}
		h.removeNodeLinks(n)
		delete(h.nodes, id)
		if levelNodes, ok := h.nodesByLevel[n.Level]; ok {
			delete(levelNodes, id)
			if len(levelNodes) == 0 {
				delete(h.nodesByLevel, n.Level)
			}
		}
	}

	// Clean up links in remaining nodes.
	for _, n := range h.nodes {
		for L, neighbors := range n.Links {
			newNeighbors := make([]*node, 0, len(neighbors))
			for _, neighbor := range neighbors {
				if _, exists := h.nodes[neighbor.ID]; exists {
					newNeighbors = append(newNeighbors, neighbor)
				}
			}
			n.Links[L] = newNeighbors
		}
	}
	// Update the entry point and the maximum level.
	h.entryPoint = nil
	h.maxLevel = -1
	for _, n := range h.nodes {
		if h.entryPoint == nil || n.Level > h.entryPoint.Level {
			h.entryPoint = n
			h.maxLevel = n.Level
		}
	}
	return nil
}

// BulkUpdate updates multiple nodes with new vectors.
func (h *Index) BulkUpdate(updates map[int][]float32) error {
	// Normalize vectors in batch when the metric requires it.
	if h.metric.Normalizes() {
		var vecs [][]float32
		for _, vector := range updates {
			if len(vector) != h.dimension {
				return fmt.Errorf("vector dimension %d does not match index dimension %d",
					len(vector), h.dimension)
			}
			vecs = append(vecs, vector)
		}
		core.NormalizeBatch(vecs)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Unlink and reinsert only the nodes being updated, as Update does.
	for id, vector := range updates {
		n, exists := h.nodes[id]
		if !exists {
			continue
		}
		if len(vector) != h.dimension {
			return fmt.Errorf("vector dimension %d does not match index dimension %d for id %d",
				len(vector), h.dimension, id)
		}
		if err := h.reinsertLocked(n, vector); err != nil {
			return err
		}
	}
	return nil
}

// Search finds the k-nearest neighbors of a given query vector.
func (h *Index) Search(query []float32, k int) ([]core.Neighbor, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if len(query) != h.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d",
			len(query), h.dimension)
	}
	if h.entryPoint == nil {
		return nil, errors.New("index is empty")
	}

	// Copy query to avoid modifying the original vector.
	queryCopy := make([]float32, len(query))
	copy(queryCopy, query)
	if h.metric.Normalizes() {
		core.NormalizeVector(queryCopy)
	}
	query = queryCopy

	// Greedy search down from the top layer.
	current := h.entryPoint
	for L := h.maxLevel; L > 0; L-- {
		changed := true
		for changed {
			changed = false
			for _, neighbor := range current.Links[L] {
				distNeighbor, err := h.metric.Rank(query, neighbor.Vector)
				if err != nil {
					return nil, err
				}
				distCurrent, err := h.metric.Rank(query, current.Vector)
				if err != nil {
					return nil, err
				}
				if distNeighbor < distCurrent {
					current = neighbor
					changed = true
				}
			}
		}
	}
	// Search in the base layer (level 0) for candidates.
	candidates, err := h.searchLayer(query, current, 0, h.ef, h.metric.Rank, nil)
	if err != nil {
		return nil, err
	}
	if len(candidates) < k {
		// Use fallback to gather more candidates if needed.
		h.fallbackSearches.Add(1)

		candidateIDs := make(map[int]bool)
		for _, c := range candidates {
			candidateIDs[c.node.ID] = true
		}

		// The scan must gather k nearest non-candidates, not only the
		// shortfall. With only the shortfall, the union holds exactly k
		// entries and a poor graph candidate can never be displaced, so the
		// search would return a wrong set despite reading every vector.
		fallbackSize := k
		var keys []int
		for id := range h.nodes {
			keys = append(keys, id)
		}
		sort.Ints(keys)
		nodesSlice := make([]*node, 0, len(h.nodes))
		for _, id := range keys {
			n := h.nodes[id]
			if candidateIDs[n.ID] {
				continue
			}
			nodesSlice = append(nodesSlice, n)
		}

		// Skip the scan when every node is already a candidate. This keeps
		// the worker count from being clamped to zero.
		if len(nodesSlice) == 0 {
			if k > len(candidates) {
				k = len(candidates)
			}
			results := make([]core.Neighbor, k)
			for i := 0; i < k; i++ {
				results[i] = core.Neighbor{
					ID:       candidates[i].node.ID,
					Distance: h.metric.FromRank(candidates[i].dist),
				}
			}
			return results, nil
		}

		numWorkers := runtime.NumCPU()
		if numWorkers > len(nodesSlice) {
			numWorkers = len(nodesSlice)
		}
		chunkSize := (len(nodesSlice) + numWorkers - 1) / numWorkers
		resultsCh := make(chan candidateMaxHeap, numWorkers)
		errsCh := make(chan error, numWorkers)
		var wg sync.WaitGroup

		// Run the fallback search in parallel. The chunk size is rounded up,
		// so the later workers can start past the end of the slice. Those
		// workers have no work left.
		for i := 0; i < numWorkers; i++ {
			start := i * chunkSize
			if start >= len(nodesSlice) {
				break
			}
			end := start + chunkSize
			if end > len(nodesSlice) {
				end = len(nodesSlice)
			}
			wg.Add(1)
			go func(nodesChunk []*node) {
				defer wg.Done()
				localHeap := candidateMaxHeap{}
				heap.Init(&localHeap)
				for _, n := range nodesChunk {
					d, err := h.metric.Rank(query, n.Vector)
					if err != nil {
						errsCh <- err
						return
					}
					cand := candidate{n, d}
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
		close(errsCh)

		for err := range errsCh {
			if err != nil {
				return nil, err // Return the first error encountered
			}
		}

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
		results[i] = core.Neighbor{
			ID:       candidates[i].node.ID,
			Distance: h.metric.FromRank(candidates[i].dist),
		}
	}
	return results, nil
}

// Stats returns simple statistics about the index.
func (h *Index) Stats() core.IndexStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	count := len(h.nodes)
	stats := core.IndexStats{
		Count:            count,
		Dimension:        h.dimension,
		Distance:         h.metric.Name(),
		FallbackSearches: h.fallbackSearches.Load(),
	}
	return stats
}

// Save writes the index to the given writer using gob encoding. The lock
// is taken by GobEncode. Taking it here as well would deadlock when a
// writer queues between the two read lock acquisitions.
func (h *Index) Save(w io.Writer) error {
	enc := gob.NewEncoder(w)
	if err := enc.Encode(h); err != nil {
		return err
	}
	return nil
}

// Load reads the index from the given reader using gob decoding.
func (h *Index) Load(r io.Reader) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	dec := gob.NewDecoder(r)
	if err := dec.Decode(h); err != nil {
		return err
	}
	return nil
}

// Check interface compliance at compile time.
var _ core.Index = (*Index)(nil)
var _ core.BulkIndex = (*Index)(nil)

// init registers types for gob encoding.
func init() {
	gob.Register(serializedIndex{})
	gob.Register(serializedNode{})
	gob.Register(&Index{})
	gob.Register(&node{})
}

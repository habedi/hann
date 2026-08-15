package hnsw

import "fmt"

// Test hooks. The tests live in the package hnsw_test, so the internals they
// need are re-exported here for the test binary only.
var LevelFromUnit = levelFromUnit

// MaxLevelCap mirrors maxLevelCap for the level tests.
const MaxLevelCap = maxLevelCap

// AuditGraph checks the structural invariants of the graph and returns a
// description of every violation: neighbor lists within M, no duplicate
// links, no links to nodes outside the index, a reverse record for every
// link, and an entry point that is a stored node at the maximum level.
// Extra reverse records without a matching link are tolerated: a trim that
// interleaves with the second half of an edge insertion can leave one
// behind, and every consumer of reverse links handles that.
func AuditGraph(h *Index) []string {
	h.mu.RLock()
	defer h.mu.RUnlock()
	var problems []string
	for id, n := range h.nodes {
		for level, links := range n.Links {
			if len(links) > h.m {
				problems = append(problems,
					fmt.Sprintf("node %d level %d has %d links, cap %d", id, level, len(links), h.m))
			}
			seen := make(map[int]struct{}, len(links))
			for _, nb := range links {
				if _, dup := seen[nb.ID]; dup {
					problems = append(problems,
						fmt.Sprintf("node %d level %d links to %d twice", id, level, nb.ID))
				}
				seen[nb.ID] = struct{}{}
				if h.nodes[nb.ID] != nb {
					problems = append(problems,
						fmt.Sprintf("node %d level %d links to missing node %d", id, level, nb.ID))
				}
				if !containsNode(nb.ReverseLinks[level], n) {
					problems = append(problems,
						fmt.Sprintf("edge %d->%d at level %d has no reverse record", id, nb.ID, level))
				}
			}
		}
	}
	if h.entryPoint != nil {
		if h.nodes[h.entryPoint.ID] != h.entryPoint {
			problems = append(problems, "entry point is not a stored node")
		}
		if h.entryPoint.Level != h.maxLevel {
			problems = append(problems,
				fmt.Sprintf("entry point level %d, maxLevel %d", h.entryPoint.Level, h.maxLevel))
		}
	} else if len(h.nodes) > 0 {
		problems = append(problems, "no entry point despite stored nodes")
	}
	return problems
}

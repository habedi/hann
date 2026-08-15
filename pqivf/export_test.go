package pqivf

// Test hooks. The tests live in package pqivf_test, so the internals they
// need are re-exported here for the test binary only.
var AssignPoints = assignPoints

// EncodeVector exposes encodeVector for the differential encode test.
func EncodeVector(pq *Index, vector []float32, cluster int) ([]int, error) {
	return pq.encodeVector(vector, cluster)
}

// Codebooks exposes the trained codebooks for the differential encode test.
func Codebooks(pq *Index) [][][]float32 { return pq.codebooks }

// CoarseCentroids exposes the coarse centroids for the differential encode
// test.
func CoarseCentroids(pq *Index) [][]float32 { return pq.coarseCentroids }

// InvertedIDs returns the entry ids of every inverted list in storage
// order, for the deterministic-order test.
func InvertedIDs(pq *Index) map[int][]int {
	out := make(map[int][]int, len(pq.invertedLists))
	for cluster, entries := range pq.invertedLists {
		ids := make([]int, len(entries))
		for i, entry := range entries {
			ids[i] = entry.ID
		}
		out[cluster] = ids
	}
	return out
}

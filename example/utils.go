package example

import (
	"fmt"
	"github.com/habedi/hann/core"
)

// FormatResults returns a formatted string of neighbor results.
// maxResults specifies how many items to include.
func FormatResults(results []core.Neighbor, maxResults int) string {
	s := ""
	limit := maxResults
	if len(results) < limit {
		limit = len(results)
	}
	for i := 0; i < limit; i++ {
		n := results[i]
		s += fmt.Sprintf("id=%d (dist=%.3f) ", n.ID, n.Distance)
	}
	return s
}

// FormatGroundTruth returns a formatted string of ground-truth neighbor results.
// maxResults specifies how many items to include.
func FormatGroundTruth(neighbors []int, distances []float64, k, maxResults int) string {
	s := ""
	limit := maxResults
	if len(neighbors) < limit {
		limit = len(neighbors)
	}
	for j := 0; j < limit; j++ {
		s += fmt.Sprintf("id=%d (dist=%.3f) ", neighbors[j], distances[j])
	}
	return s
}

// RecallAtK computes Recall@k as the fraction of all ground-truth items that appear in the top k predictions.
func RecallAtK(predicted []core.Neighbor, groundTruth []int, k int) float64 {
	if k <= 0 || len(groundTruth) == 0 {
		return 0.0
	}
	// Build a set of predicted IDs from the top k predictions.
	predSet := make(map[int]struct{})
	limit := k
	if len(predicted) < k {
		limit = len(predicted)
	}
	for i := 0; i < limit; i++ {
		predSet[predicted[i].ID] = struct{}{}
	}

	// Count ground-truth items that appear in the predictions.
	correct := 0
	for _, id := range groundTruth {
		if _, ok := predSet[id]; ok {
			correct++
		}
	}
	return float64(correct) / float64(len(groundTruth))
}

package examples

import (
	"fmt"
	"github.com/habedi/hann/core"
)

// FormatResults returns a formatted string of neighbor results.
func FormatResults(results []core.Neighbor) string {
	s := ""
	for _, n := range results {
		s += fmt.Sprintf("id=%d (dist=%.3f) ", n.ID, n.Distance)
	}
	return s
}

// FormatGroundTruth returns a formatted string of ground-truth neighbor results.
func FormatGroundTruth(neighbors []int, distances []float64, k int) string {
	s := ""
	for j := 0; j < k && j < len(neighbors); j++ {
		s += fmt.Sprintf("id=%d (dist=%.3f) ", neighbors[j], distances[j])
	}
	return s
}

// RecallAtK computes the recall@k given predicted neighbor results and ground-truth neighbor IDs.
// It returns the fraction (between 0 and 1) of ground-truth IDs that appear in the top k predictions.
func RecallAtK(predicted []core.Neighbor, groundTruth []int, k int) float64 {
	if k <= 0 || len(groundTruth) == 0 {
		return 0.0
	}
	// Build a set of predicted neighbor IDs from the top k results.
	predSet := make(map[int]struct{})
	for i := 0; i < k && i < len(predicted); i++ {
		predSet[predicted[i].ID] = struct{}{}
	}

	// Count how many ground-truth IDs are present in the predictions.
	correct := 0
	for _, id := range groundTruth {
		if _, ok := predSet[id]; ok {
			correct++
		}
	}

	return float64(correct) / float64(len(groundTruth))
}

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

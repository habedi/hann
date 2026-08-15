package example_test

import (
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
)

func TestFormatResults(t *testing.T) {
	results := []core.Neighbor{
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 1.25},
	}

	t.Run("all results", func(t *testing.T) {
		want := "id=1 (dist=0.500) id=2 (dist=1.250) "
		if got := example.FormatResults(results, 2); got != want {
			t.Errorf("expected %q, got %q", want, got)
		}
	})

	t.Run("maxResults smaller than results", func(t *testing.T) {
		want := "id=1 (dist=0.500) "
		if got := example.FormatResults(results, 1); got != want {
			t.Errorf("expected %q, got %q", want, got)
		}
	})

	t.Run("maxResults larger than results", func(t *testing.T) {
		want := "id=1 (dist=0.500) id=2 (dist=1.250) "
		if got := example.FormatResults(results, 10); got != want {
			t.Errorf("expected %q, got %q", want, got)
		}
	})

	t.Run("empty results", func(t *testing.T) {
		if got := example.FormatResults(nil, 5); got != "" {
			t.Errorf("expected an empty string, got %q", got)
		}
	})
}

func TestFormatGroundTruth(t *testing.T) {
	neighbors := []int{7, 8}
	distances := []float64{0.5, 1.25}

	t.Run("all neighbors", func(t *testing.T) {
		want := "id=7 (dist=0.500) id=8 (dist=1.250) "
		if got := example.FormatGroundTruth(neighbors, distances, 2, 2); got != want {
			t.Errorf("expected %q, got %q", want, got)
		}
	})

	t.Run("maxResults smaller than neighbors", func(t *testing.T) {
		want := "id=7 (dist=0.500) "
		if got := example.FormatGroundTruth(neighbors, distances, 2, 1); got != want {
			t.Errorf("expected %q, got %q", want, got)
		}
	})

	t.Run("maxResults larger than neighbors", func(t *testing.T) {
		want := "id=7 (dist=0.500) id=8 (dist=1.250) "
		if got := example.FormatGroundTruth(neighbors, distances, 2, 10); got != want {
			t.Errorf("expected %q, got %q", want, got)
		}
	})

	t.Run("empty neighbors", func(t *testing.T) {
		if got := example.FormatGroundTruth(nil, nil, 2, 5); got != "" {
			t.Errorf("expected an empty string, got %q", got)
		}
	})
}

func TestRecallAtK(t *testing.T) {
	groundTruth := make([]int, 100)
	for i := range groundTruth {
		groundTruth[i] = i
	}

	t.Run("perfect top-k predictions", func(t *testing.T) {
		predicted := make([]core.Neighbor, 10)
		for i := range predicted {
			predicted[i] = core.Neighbor{ID: i}
		}
		if got := example.RecallAtK(predicted, groundTruth, 10); got != 1.0 {
			t.Errorf("expected recall 1.0, got %v", got)
		}
	})

	t.Run("half of top-k predictions correct", func(t *testing.T) {
		predicted := make([]core.Neighbor, 10)
		for i := 0; i < 5; i++ {
			predicted[i] = core.Neighbor{ID: i}
		}
		for i := 5; i < 10; i++ {
			predicted[i] = core.Neighbor{ID: 1000 + i}
		}
		if got := example.RecallAtK(predicted, groundTruth, 10); got != 0.5 {
			t.Errorf("expected recall 0.5, got %v", got)
		}
	})

	t.Run("k larger than ground truth", func(t *testing.T) {
		shortTruth := []int{0, 1, 2, 3}
		predicted := make([]core.Neighbor, 10)
		for i := range predicted {
			predicted[i] = core.Neighbor{ID: i}
		}
		if got := example.RecallAtK(predicted, shortTruth, 10); got != 1.0 {
			t.Errorf("expected recall 1.0 with short ground truth, got %v", got)
		}
	})

	t.Run("k equals ground truth length", func(t *testing.T) {
		predicted := make([]core.Neighbor, 100)
		for i := range predicted {
			predicted[i] = core.Neighbor{ID: i}
		}
		if got := example.RecallAtK(predicted, groundTruth, 100); got != 1.0 {
			t.Errorf("expected recall 1.0, got %v", got)
		}
	})

	t.Run("fewer predictions than k", func(t *testing.T) {
		predicted := []core.Neighbor{{ID: 0}, {ID: 1}}
		if got := example.RecallAtK(predicted, groundTruth, 10); got != 0.2 {
			t.Errorf("expected recall 0.2 with two predictions, got %v", got)
		}
	})

	t.Run("invalid arguments", func(t *testing.T) {
		if got := example.RecallAtK(nil, groundTruth, 0); got != 0.0 {
			t.Errorf("expected recall 0.0 for k=0, got %v", got)
		}
		if got := example.RecallAtK(nil, nil, 10); got != 0.0 {
			t.Errorf("expected recall 0.0 for empty ground truth, got %v", got)
		}
	})
}

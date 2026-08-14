package example_test

import (
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
)

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

	t.Run("invalid arguments", func(t *testing.T) {
		if got := example.RecallAtK(nil, groundTruth, 0); got != 0.0 {
			t.Errorf("expected recall 0.0 for k=0, got %v", got)
		}
		if got := example.RecallAtK(nil, nil, 10); got != 0.0 {
			t.Errorf("expected recall 0.0 for empty ground truth, got %v", got)
		}
	})
}

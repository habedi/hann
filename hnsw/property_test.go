package hnsw_test

import (
	"fmt"
	"testing"

	"github.com/habedi/hann/internal/testutil"
)

func TestHNSWIndex_PropertyOps(t *testing.T) {
	for _, seed := range []int64{1, 2, 3, 42, 12345} {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			testutil.RunPropertyOps(t, hnswFactory(), 16, seed, 400)
		})
	}
}

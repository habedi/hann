package rpt_test

import (
	"fmt"
	"testing"

	"github.com/habedi/hann/internal/testutil"
)

// TestRPTIndex_PropertyOps runs random operation sequences against a
// brute-force model, checking counts, error paths, search membership, exact
// distances, sorted results, and save and load round-trips.
func TestRPTIndex_PropertyOps(t *testing.T) {
	for _, seed := range []int64{1, 2, 3, 4} {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			testutil.RunPropertyOps(t, rptFactory(t, 16), 16, seed, 400)
		})
	}
}

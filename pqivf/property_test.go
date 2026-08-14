package pqivf_test

import (
	"fmt"
	"testing"

	"github.com/habedi/hann/internal/testutil"
)

func TestPQIVF_PropertyOps(t *testing.T) {
	// The runs stay at 140 ops per seed, because the op mix in
	// RunPropertyOps adds ids faster than it deletes them, and its freeID
	// helper loops forever once all 64 ids in the id space are live, which
	// happens near op 160. Five seeds give 700 ops in total.
	for _, seed := range []int64{1, 2, 3, 4, 5} {
		seed := seed
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			testutil.RunPropertyOps(t, pqivfFactory(t), 16, seed, 140)
		})
	}
}

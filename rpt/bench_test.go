package rpt_test

import (
	"testing"

	"github.com/habedi/hann/internal/testutil"
	"github.com/habedi/hann/rpt"
)

const (
	benchSeed     = 42
	benchN        = 2000
	benchDim      = 32
	benchClusters = 8
	benchK        = 10
	benchQueries  = 100
)

// BenchmarkBuild measures constructing an index, bulk-adding the full dataset,
// and one search, so it reports whole-build time. The search is included
// because the tree is built lazily on the first search after a mutation.
func BenchmarkBuild(b *testing.B) {
	data := testutil.ClusteredData(benchSeed, benchN, benchDim, benchClusters)
	query := testutil.Queries(benchSeed+1, data, 1)[0]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := rpt.New(benchDim)
		if err != nil {
			b.Fatal(err)
		}
		if err := idx.BulkAdd(data); err != nil {
			b.Fatal(err)
		}
		if _, err := idx.Search(query, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSearch measures Search on an index built once in setup. A warm-up
// search runs before the timer starts, so the lazy tree build is not counted.
// The timed loop cycles through precomputed queries by iteration index.
func BenchmarkSearch(b *testing.B) {
	data := testutil.ClusteredData(benchSeed, benchN, benchDim, benchClusters)
	idx, err := rpt.New(benchDim)
	if err != nil {
		b.Fatal(err)
	}
	if err := idx.BulkAdd(data); err != nil {
		b.Fatal(err)
	}
	queries := testutil.Queries(benchSeed+1, data, benchQueries)
	if _, err := idx.Search(queries[0], benchK); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Search(queries[i%len(queries)], benchK); err != nil {
			b.Fatal(err)
		}
	}
}

package pqivf_test

import (
	"testing"

	"github.com/habedi/hann/internal/testutil"
	"github.com/habedi/hann/pqivf"
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
// and training, so it reports whole-build time. Train is included because an
// untrained PQIVF index holds its vectors unquantized and cannot serve a
// search.
func BenchmarkBuild(b *testing.B) {
	data := testutil.ClusteredData(benchSeed, benchN, benchDim, benchClusters)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := pqivf.New(benchDim)
		if err != nil {
			b.Fatal(err)
		}
		if err := idx.BulkAdd(data); err != nil {
			b.Fatal(err)
		}
		if err := idx.Train(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSearch measures Search on an index built and trained once in
// setup, cycling through precomputed queries by iteration index.
func BenchmarkSearch(b *testing.B) {
	data := testutil.ClusteredData(benchSeed, benchN, benchDim, benchClusters)
	idx, err := pqivf.New(benchDim)
	if err != nil {
		b.Fatal(err)
	}
	if err := idx.BulkAdd(data); err != nil {
		b.Fatal(err)
	}
	if err := idx.Train(); err != nil {
		b.Fatal(err)
	}
	queries := testutil.Queries(benchSeed+1, data, benchQueries)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Search(queries[i%len(queries)], benchK); err != nil {
			b.Fatal(err)
		}
	}
}

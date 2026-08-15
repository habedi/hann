package hnsw_test

import (
	"testing"

	"github.com/habedi/hann/hnsw"
	"github.com/habedi/hann/internal/testutil"
)

const (
	benchSeed     = 42
	benchN        = 2000
	benchDim      = 32
	benchClusters = 8
	benchK        = 10
	benchQueries  = 100
)

// BenchmarkBuild measures building an index and bulk-adding the full
// dataset. It reports whole-build time rather than per-insert time.
func BenchmarkBuild(b *testing.B) {
	data := testutil.ClusteredData(benchSeed, benchN, benchDim, benchClusters)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := hnsw.New(benchDim)
		if err != nil {
			b.Fatal(err)
		}
		if err := idx.BulkAdd(data); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSearch measures Search on an index built once in setup. It
// cycles through precomputed queries by iteration index.
func BenchmarkSearch(b *testing.B) {
	data := testutil.ClusteredData(benchSeed, benchN, benchDim, benchClusters)
	idx, err := hnsw.New(benchDim)
	if err != nil {
		b.Fatal(err)
	}
	if err := idx.BulkAdd(data); err != nil {
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

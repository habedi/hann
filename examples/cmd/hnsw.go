package main

import (
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/examples"
	"github.com/habedi/hann/hnsw"
)

func main() {
	// Dataset to load
	dataset := "fashion-mnist-784-euclidean"
	root := "examples/data/nearest-neighbors-datasets"
	datasetPath := filepath.Join(root, dataset)

	fmt.Println("Loading dataset:", dataset)
	start := time.Now()

	// Create HNSW index with dimension and parameters
	dimension := 784
	M := 16
	ef := 64
	index := hnsw.NewHNSW(dimension, M, ef)

	// Load training vectors into index and get test set + ground truth
	testVectors, gtNeighbors, gtDistances, err := examples.LoadDataset(index, datasetPath)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	stats := index.Stats()
	fmt.Printf("Indexed %d vectors (%d dimensions) in %.2fs\n", stats.Count, stats.Dimension,
		time.Since(start).Seconds())

	// Run k-NN search on the first few test queries
	k := 10
	numQueries := 5
	fmt.Printf("\nRunning k-NN search (k=%d) on first %d test queries...\n\n", k, numQueries)

	for i := 0; i < numQueries && i < len(testVectors); i++ {
		query := testVectors[i]
		results, err := index.Search(query, k, core.Euclidean)
		if err != nil {
			log.Fatalf("Search error on query %d: %v", i, err)
		}

		fmt.Printf("Query #%d\n", i)
		fmt.Printf("  Predicted:     ")
		for _, n := range results {
			fmt.Printf("%d (%.3f) ", n.ID, n.Distance)
		}
		fmt.Printf("\n  Ground-truth:  ")
		for j := 0; j < k && j < len(gtNeighbors[i]); j++ {
			fmt.Printf("%d (%.3f) ", gtNeighbors[i][j], gtDistances[i][j])
		}
		fmt.Println()
	}
}

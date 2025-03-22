package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/examples"
	"github.com/habedi/hann/hnsw"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	// Logging configuration is handled in core/log_config.go.
	// Here we just set the console writer.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Dataset configuration.
	dataset := "fashion-mnist-784-euclidean"
	root := "examples/data/nearest-neighbors-datasets"
	datasetPath := filepath.Join(root, dataset)

	// Use fmt.Printf to output summary information regardless of logging level.
	fmt.Printf("Loading dataset: %s\n", dataset)
	start := time.Now()

	// Create HNSW index.
	dimension := 784
	M := 16
	ef := 64
	index := hnsw.NewHNSW(dimension, M, ef)
	fmt.Printf("Created HNSW index (dimension=%d, M=%d, ef=%d)\n", dimension, M, ef)

	// Load training vectors and BulkAdd.
	trainingVectors, err := examples.LoadTrainingVectors(datasetPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load training vectors")
	}
	if err := index.BulkAdd(trainingVectors); err != nil {
		log.Fatal().Err(err).Msg("BulkAdd failed")
	}

	// Load test dataset.
	testVectors, gtNeighbors, gtDistances, err := examples.LoadTestDataset(datasetPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load test dataset")
	}

	stats := index.Stats()
	fmt.Printf("Indexed %d vectors (%d dimensions) in %.2fs; index size: %.2f mb\n",
		stats.Count, stats.Dimension, time.Since(start).Seconds(), float64(stats.Size)/1e6)

	// Run k-NN search on the first few test queries.
	k := 5
	numQueries := 5
	fmt.Printf("Running k-NN search (k=%d) on first %d test queries\n", k, numQueries)

	for i := 0; i < numQueries && i < len(testVectors); i++ {
		query := testVectors[i]
		results, err := index.Search(query, k, core.Euclidean)
		if err != nil {
			log.Fatal().Err(err).Msgf("Search error on query %d", i)
		}
		// Print only the summary results.
		fmt.Printf("Query #%d:\n", i+1)
		fmt.Printf(" -> Predicted:     %s\n", formatResults(results))
		fmt.Printf(" -> Ground-truth:  %s\n", formatGroundTruth(gtNeighbors[i], gtDistances[i], k))
	}
}

// formatResults returns a formatted string of neighbor results.
func formatResults(results []core.Neighbor) string {
	s := ""
	for _, n := range results {
		s += fmt.Sprintf("id=%d (dist=%.3f) ", n.ID, n.Distance)
	}
	return s
}

// formatGroundTruth returns a formatted string of ground-truth neighbor results.
func formatGroundTruth(neighbors []int, distances []float64, k int) string {
	s := ""
	for j := 0; j < k && j < len(neighbors); j++ {
		s += fmt.Sprintf("id=%d (dist=%.3f) ", neighbors[j], distances[j])
	}
	return s
}

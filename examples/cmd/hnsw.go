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
	// Configure zerolog to write human-readable logs.
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Dataset configuration.
	dataset := "fashion-mnist-784-euclidean"
	root := "examples/data/nearest-neighbors-datasets"
	datasetPath := filepath.Join(root, dataset)

	log.Info().Msgf("Loading dataset: %s", dataset)
	start := time.Now()

	// Create HNSW index with dimension and parameters.
	dimension := 784
	M := 16
	ef := 64
	index := hnsw.NewHNSW(dimension, M, ef)
	log.Info().Msgf("Created HNSW index (dimension=%d, M=%d, ef=%d)", dimension, M, ef)

	// Load training vectors into index and get test set + ground truth.
	testVectors, gtNeighbors, gtDistances, err := examples.LoadDataset(index, datasetPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load dataset")
	}

	stats := index.Stats()
	log.Info().Msgf("Indexed %d vectors (%d dimensions) in %.2fs", stats.Count,
		stats.Dimension, time.Since(start).Seconds())

	// Run k-NN search on the first few test queries.
	k := 10
	numQueries := 5
	log.Info().Msgf("Running k-NN search (k=%d) on first %d test queries", k, numQueries)

	for i := 0; i < numQueries && i < len(testVectors); i++ {
		query := testVectors[i]
		results, err := index.Search(query, k, core.Euclidean)
		if err != nil {
			log.Fatal().Err(err).Msgf("Search error on query %d", i)
		}

		log.Info().Msgf("Query #%d:", i)
		log.Info().Msgf("  Predicted:     %s", formatResults(results))
		log.Info().Msgf("  Ground-truth:  %s", formatGroundTruth(gtNeighbors[i],
			gtDistances[i], k))
	}
}

// formatResults returns a formatted string of neighbor results.
func formatResults(results []core.Neighbor) string {
	s := ""
	for _, n := range results {
		s += fmt.Sprintf("%d (%.3f) ", n.ID, n.Distance)
	}
	return s
}

// formatGroundTruth returns a formatted string of ground-truth neighbor results.
func formatGroundTruth(neighbors []int, distances []float64, k int) string {
	s := ""
	for j := 0; j < k && j < len(neighbors); j++ {
		s += fmt.Sprintf("%d (%.3f) ", neighbors[j], distances[j])
	}
	return s
}

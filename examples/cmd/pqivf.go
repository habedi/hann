package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/examples"
	"github.com/habedi/hann/pqivf"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	// Logging configuration (see core/log_config.go for environment-based settings)
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Dataset configuration.
	dataset := "fashion-mnist-784-euclidean"
	root := "examples/data/nearest-neighbors-datasets"
	datasetPath := filepath.Join(root, dataset)

	fmt.Printf("Loading dataset: %s\n", dataset)
	start := time.Now()

	// Create a PQIVF index.
	// Adjust these parameters (dimension, coarseK, numSubquantizers) as needed.
	dimension := 784
	coarseK := 16
	numSubquantizers := 8
	index := pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers)
	fmt.Printf("Created PQIVF index (dimension=%d, coarseK=%d, numSubquantizers=%d)\n", dimension, coarseK, numSubquantizers)

	// Load training vectors (into the index) from train.csv.
	trainingVectors, err := examples.LoadTrainingVectors(datasetPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load training vectors")
	}
	if err := index.BulkAdd(trainingVectors); err != nil {
		log.Fatal().Err(err).Msg("BulkAdd failed")
	}

	// Load test dataset (test vectors, ground-truth neighbors, and distances).
	testVectors, gtNeighbors, gtDistances, err := examples.LoadTestDataset(datasetPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load test dataset")
	}

	stats := index.Stats()
	fmt.Printf("Indexed %d vectors (%d dimensions) in %.2fs; index size: %.2f mb\n",
		stats.Count, stats.Dimension, time.Since(start).Seconds(), float64(stats.Size)/1e6)

	// Run kNN search on the first few test queries.
	k := 5
	numQueries := 5
	fmt.Printf("Running kNN search (k=%d) on first %d test queries\n", k, numQueries)

	for i := 0; i < numQueries && i < len(testVectors); i++ {
		query := testVectors[i]
		results, err := index.Search(query, k, core.Euclidean)
		if err != nil {
			log.Fatal().Err(err).Msgf("Search error on query %d", i)
		}
		// Compute Recall@k.
		recall := examples.RecallAtK(results, gtNeighbors[i], k)

		// Print only the summary results.
		fmt.Printf("Query #%d:\n", i+1)
		fmt.Printf(" -> Predicted:     %s\n", examples.FormatResults(results))
		fmt.Printf(" -> Ground-truth:  %s\n", examples.FormatGroundTruth(gtNeighbors[i],
			gtDistances[i], k))
		fmt.Printf(" -> Recall@%d:     %.2f\n", k, recall)
	}
}

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/examples"
	"github.com/habedi/hann/rpt"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	// Logging configuration (set via environment variables in core/log_config.go)
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Dataset configuration.
	dataset := "fashion-mnist-784-euclidean"
	root := "examples/data/nearest-neighbors-datasets"
	datasetPath := filepath.Join(root, dataset)

	fmt.Printf("Loading dataset: %s\n", dataset)
	start := time.Now()

	// Create an RPT index.
	dimension := 784
	index := rpt.NewRPTIndex(dimension)
	fmt.Printf("Created RPT index (dimension=%d)\n", dimension)

	// Load training vectors (into the index) from train.csv.
	trainingVectors, err := examples.LoadTrainingVectors(datasetPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load training vectors")
	}
	// For RPT, use BulkAdd to add all training vectors.
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

		// Max number of results to display.
		maxResults := 10

		// Compute Recall@k.
		recall := examples.RecallAtK(results, gtNeighbors[i], k)

		// Print only the summary results.
		fmt.Printf("Query #%d:\n", i+1)
		fmt.Printf(" -> Predicted:     %s\n", examples.FormatResults(results, maxResults))
		fmt.Printf(" -> Ground-truth:  %s\n", examples.FormatGroundTruth(gtNeighbors[i],
			gtDistances[i], k, maxResults))
		fmt.Printf(" -> Recall@%d:     %.2f\n", k, recall)
	}
}

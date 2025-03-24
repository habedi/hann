package example

import (
	"fmt"
	"path/filepath"
	"time"

	"github.com/habedi/hann/core"
	"github.com/rs/zerolog/log"
)

// IndexFactory is a function that creates a new index.
type IndexFactory func() core.Index

// RunDataset loads the dataset from the specified directory, builds the index using the provided factory,
// and runs kNN search on the first numQueries test queries. It prints the predicted results,
// ground-truth, and computes Recall@k.
func RunDataset(factory IndexFactory, dataset, root string, k, numQueries, maxResults int) {
	datasetPath := filepath.Join(root, dataset)
	fmt.Printf("Loading dataset: %s\n", dataset)
	start := time.Now()

	// Create the index.
	index := factory()
	fmt.Printf("Created index: %T\n", index)

	// Load training vectors and add them to the index.
	trainingVectors, err := LoadTrainingVectors(datasetPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load training vectors")
	}
	log.Info().Msgf("Loaded %d training vectors", len(trainingVectors))
	if err := index.BulkAdd(trainingVectors); err != nil {
		log.Fatal().Err(err).Msg("BulkAdd failed")
	}

	// Load test dataset.
	testVectors, gtNeighbors, gtDistances, err := LoadTestDataset(datasetPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load test dataset")
	}
	log.Info().Msgf("Loaded %d test vectors", len(testVectors))

	stats := index.Stats()
	fmt.Printf("Indexed %d vectors (%d dimensions) in %.2fs; distance: %s\n",
		stats.Count, stats.Dimension, time.Since(start).Seconds(), stats.Distance)

	fmt.Printf("Running kNN search (k=%d) on first %d test queries\n", k, numQueries)
	for i := 0; i < numQueries && i < len(testVectors); i++ {
		query := testVectors[i]
		results, err := index.Search(query, k)
		if err != nil {
			log.Fatal().Err(err).Msgf("Search error on query %d", i)
		}
		recall := RecallAtK(results, gtNeighbors[i], k)
		fmt.Printf("Query #%d:\n", i+1)
		fmt.Printf(" -> Predicted:     %s\n", FormatResults(results, maxResults))
		fmt.Printf(" -> Ground-truth:  %s\n", FormatGroundTruth(gtNeighbors[i],
			gtDistances[i], k, maxResults))
		fmt.Printf(" -> Recall@%d:     %.2f\n", k, recall)
	}
}

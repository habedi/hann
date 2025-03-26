package example

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/habedi/hann/core"
	"github.com/rs/zerolog/log"
	"github.com/schollz/progressbar/v3"
)

// IndexFactory is a function that creates a new index.
type IndexFactory func() core.Index

// QueryResult holds the results for a single query.
type QueryResult struct {
	idx         int
	recall      float64
	duration    time.Duration
	predicted   string
	groundTruth string
}

// RunDataset loads the dataset, builds the index using the provided factory,
// and runs kNN queries on a subset of test queries. If numQueries is negative
// or exceeds the number of available test vectors, all test vectors are used.
// It prints predicted results, ground-truth (if not benchmarking), and computes
// Recall@k along with per-query response times, average response time, and overall runtime.
// When benchmarking, a progress bar is displayed.
// The number of worker threads is read from the HANN_BENCH_NTRD environment variable.
func RunDataset(factory IndexFactory, dataset, root string, k, numQueries, maxResults int) {
	datasetPath := filepath.Join(root, dataset)
	fmt.Printf("Loading dataset: %s\n", dataset)
	overallStart := time.Now()

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
		stats.Count, stats.Dimension, time.Since(overallStart).Seconds(), stats.Distance)

	// Activate benchmark mode if numQueries is negative or too high.
	benchmarkMode := false
	if numQueries < 0 || numQueries > len(testVectors) {
		numQueries = len(testVectors)
		benchmarkMode = true
	}

	// Get the number of threads from HANN_BENCH_NTRD.
	threads := 1
	if env := os.Getenv("HANN_BENCH_NTRD"); env != "" {
		if t, err := strconv.Atoi(env); err == nil && t > 0 {
			threads = t
			log.Info().Msgf("Using %d threads used for benchmarking", threads)
		}
	}

	fmt.Printf("Running kNN queries (k=%d) on %d test vectors using %d threads\n", k, numQueries, threads)

	var totalRecall float64
	var totalQueryTime time.Duration

	// Pre-allocate a slice to hold query results.
	resultsSlice := make([]QueryResult, numQueries)

	// Set up a progress bar if in benchmark mode.
	var bar *progressbar.ProgressBar
	if benchmarkMode {
		bar = progressbar.Default(int64(numQueries))
	}

	// Create a channel to feed query indices.
	tasks := make(chan int, numQueries)
	var wg sync.WaitGroup

	// Worker function: processes queries from the task channel.
	worker := func() {
		defer wg.Done()
		for idx := range tasks {
			query := testVectors[idx]
			startQuery := time.Now()
			res, err := index.Search(query, k)
			if err != nil {
				log.Fatal().Err(err).Msgf("Search error on query %d", idx)
			}
			duration := time.Since(startQuery)
			recall := RecallAtK(res, gtNeighbors[idx], k)

			var predicted, groundTruth string
			if !benchmarkMode {
				predicted = FormatResults(res, maxResults)
				groundTruth = FormatGroundTruth(gtNeighbors[idx], gtDistances[idx], k, maxResults)
			}

			resultsSlice[idx] = QueryResult{
				idx:         idx,
				recall:      recall,
				duration:    duration,
				predicted:   predicted,
				groundTruth: groundTruth,
			}

			if benchmarkMode && bar != nil {
				err := bar.Add(1)
				if err != nil {
					return
				}
			}
		}
	}

	// Start worker goroutines.
	wg.Add(threads)
	for i := 0; i < threads; i++ {
		go worker()
	}

	// Feed query indices into the task channel.
	for i := 0; i < numQueries; i++ {
		tasks <- i
	}
	close(tasks)
	wg.Wait()

	// Aggregate the results.
	for _, res := range resultsSlice {
		totalRecall += res.recall
		totalQueryTime += res.duration
	}

	avgRecall := totalRecall / float64(numQueries)
	avgResponseTime := totalQueryTime / time.Duration(numQueries)

	// If not benchmarking, print each query's details.
	if !benchmarkMode {
		for i, res := range resultsSlice {
			fmt.Printf("Query #%d:\n", i+1)
			fmt.Printf(" -> Predicted:     %s\n", res.predicted)
			fmt.Printf(" -> Ground-truth:  %s\n", res.groundTruth)
			fmt.Printf(" -> Recall@%d:     %.2f, Response time: %v\n", k, res.recall, res.duration)
		}
	}

	fmt.Printf("Average Recall@%d over %d queries: %.2f\n", k, numQueries, avgRecall)
	fmt.Printf("Average query response time: %v\n", avgResponseTime)
	fmt.Printf("Overall runtime: %v\n", time.Since(overallStart))
}

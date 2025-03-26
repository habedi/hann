//go:build ignore
// +build ignore

package main

import (
	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/pqivf"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"net/http"
	_ "net/http/pprof"
	"os"
)

func main() {
	// Set the logger to output to the console.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Start the pprof HTTP server on port 6060.
	// This will expose profiling endpoints at /debug/pprof/
	go func() {
		log.Info().Msg("Starting pprof server on :6060")
		if err := http.ListenAndServe("localhost:6060", nil); err != nil {
			log.Error().Err(err).Msg("pprof server failed")
		}
	}()

	// Benchmarking PQIVF index with FashionMNIST and SIFT datasets
	BenchPQIVFIndexFashionMNIST()
	BenchPQIVFIndexSIFT()
}

func BenchPQIVFIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		coarseK := 16
		numSubquantizers := 8
		pqK := 256
		kMeansIters := 10
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

func BenchPQIVFIndexSIFT() {
	factory := func() core.Index {
		dimension := 128
		coarseK := 16
		numSubquantizers := 8
		pqK := 256
		kMeansIters := 10
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters)
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

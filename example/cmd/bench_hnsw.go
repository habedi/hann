//go:build ignore
// +build ignore

package main

import (
	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/hnsw"
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

	// Benchmarking HNSW index with FashionMNIST and Glove datasets
	BenchHNSWIndexFashionMNIST()
	BenchHNSWIndexGlove25()
	BenchHNSWIndexGlove200()
}

func BenchHNSWIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		M := 32
		ef := 300
		distanceName := "euclidean"
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

func BenchHNSWIndexGlove25() {
	factory := func() core.Index {
		dimension := 25
		M := 16
		ef := 100
		distanceName := "cosine"
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-25-angular",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

func BenchHNSWIndexGlove200() {
	factory := func() core.Index {
		dimension := 200
		M := 16
		ef := 100
		distanceName := "cosine"
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-200-angular",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

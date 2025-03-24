//go:build ignore
// +build ignore

package main

import (
	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/pqivf"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"os"
)

func main() {
	// Set the logger to output to the console.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	PQIVFIndexFashionMNIST("euclidean")
	PQIVFIndexSIFT("squared_euclidean")
	PQIVFIndexGlove50("cosine")
}

func PQIVFIndexFashionMNIST(distanceName string) {
	factory := func() core.Index {
		dimension := 784
		coarseK := 16
		numSubquantizers := 8
		pqK := 256
		kMeansIters := 10
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters,
			core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func PQIVFIndexSIFT(distanceName string) {
	factory := func() core.Index {
		dimension := 128
		coarseK := 16
		numSubquantizers := 8
		pqK := 256
		kMeansIters := 10
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters,
			core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func PQIVFIndexGlove50(distanceName string) {
	factory := func() core.Index {
		dimension := 50
		coarseK := 16
		numSubquantizers := 8
		pqK := 256
		kMeansIters := 10
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters,
			core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-50-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

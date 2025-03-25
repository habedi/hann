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

	// Using PQIVF index with FashionMNIST and SIFT datasets
	PQIVFIndexFashionMNIST()
	PQIVFIndexSIFT()
}

func PQIVFIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		coarseK := 16
		numSubquantizers := 8
		pqK := 256
		kMeansIters := 10
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func PQIVFIndexSIFT() {
	factory := func() core.Index {
		dimension := 128
		coarseK := 16
		numSubquantizers := 8
		pqK := 256
		kMeansIters := 10
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters)
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

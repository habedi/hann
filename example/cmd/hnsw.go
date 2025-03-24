//go:build ignore
// +build ignore

package main

import (
	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/hnsw"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"os"
)

func main() {
	// Set the logger to output to the console.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	HNSWIndexFashionMNIST("euclidean")
	HNSWIndexGlove100("cosine")
}

func HNSWIndexFashionMNIST(distanceName string) {
	factory := func() core.Index {
		dimension := 784
		M := 16
		ef := 64
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func HNSWIndexGlove100(distanceName string) {
	factory := func() core.Index {
		dimension := 100
		M := 16
		ef := 64
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-100-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

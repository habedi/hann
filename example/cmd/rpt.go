//go:build ignore
// +build ignore

package main

import (
	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/rpt"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"os"
)

func main() {
	// Set the logger to output to the console
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	RPTIndexFashionMNIST("euclidean")
	RPTIndexSIFT128("squared_euclidean")
	RPTIndexGlove50("cosine")
}

func RPTIndexFashionMNIST(distanceName string) {
	factory := func() core.Index {
		dimension := 784
		return rpt.NewRPTIndex(dimension, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func RPTIndexSIFT128(distanceName string) {
	factory := func() core.Index {
		dimension := 128
		return rpt.NewRPTIndex(dimension, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func RPTIndexGlove50(distanceName string) {
	factory := func() core.Index {
		dimension := 50
		return rpt.NewRPTIndex(dimension, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-50-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

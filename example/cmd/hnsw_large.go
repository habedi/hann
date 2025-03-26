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

	// Using HNSW index with GIST and DEEP1B datasets
	HNSWIndexGIST("euclidean")
	HNSWIndexDEEP1B("cosine")
}

func HNSWIndexGIST(distanceName string) {
	factory := func() core.Index {
		dimension := 960
		M := 16
		ef := 100
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "gist-960-euclidean",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}

func HNSWIndexDEEP1B(distanceName string) {
	factory := func() core.Index {
		dimension := 96
		M := 16
		ef := 100
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "deep-image-96-angular",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}

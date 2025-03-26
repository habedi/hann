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

	// Using HNSW index with FashionMNIST and Glove datasets
	HNSWIndexFashionMNIST()
	//BenchHNSWIndexGlove25()
	HNSWIndexGlove200()
}

func HNSWIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		M := 16
		ef := 100
		distanceName := "euclidean"
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func HNSWIndexGlove25() {
	factory := func() core.Index {
		dimension := 25
		M := 16
		ef := 100
		distanceName := "cosine"
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-25-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func HNSWIndexGlove200() {
	factory := func() core.Index {
		dimension := 200
		M := 16
		ef := 100
		distanceName := "cosine"
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-200-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

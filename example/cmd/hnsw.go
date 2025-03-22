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
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	HNSWIndexFashionMNIST()
	//HNSWIndexGlove200()

}

func HNSWIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		M := 16
		ef := 64
		distanceName := "euclidean"
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func HNSWIndexGlove200() {
	factory := func() core.Index {
		dimension := 200
		M := 16
		ef := 64
		distanceName := "angular"
		return hnsw.NewHNSW(dimension, M, ef, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-200-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

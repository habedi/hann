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
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	PQIVFIndexFashionMNIST()
	//PQIVFIndexGlove200()
}

func PQIVFIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		coarseK := 16
		numSubquantizers := 8
		distanceName := "euclidean"
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers,
			core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func PQIVFIndexGlove200() {
	factory := func() core.Index {
		dimension := 200
		coarseK := 16
		numSubquantizers := 8
		distanceName := "angular"
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers,
			core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-200-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

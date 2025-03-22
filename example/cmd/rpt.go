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
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	RPTIndexFashionMNIST()
	//RPTIndexGlove200()
}

func RPTIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		distanceName := "euclidean"
		return rpt.NewRPTIndex(dimension, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func RPTIndexGlove200() {
	factory := func() core.Index {
		dimension := 200
		distanceName := "angular"
		return rpt.NewRPTIndex(dimension, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "glove-200-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

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
	// Set the logger to output to the console.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	RPTIndexGIST("euclidean")
	RPTIndexDEEP1B("cosine")
}

func RPTIndexGIST(distanceName string) {
	factory := func() core.Index {
		dimension := 960
		leafCapacity := 10
		candidateProjections := 3
		parallelThreshold := 100
		probeMargin := 0.15
		return rpt.NewRPTIndex(dimension, leafCapacity, candidateProjections, parallelThreshold,
			probeMargin, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "gist-960-euclidean",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}

func RPTIndexDEEP1B(distanceName string) {
	factory := func() core.Index {
		dimension := 96
		leafCapacity := 10
		candidateProjections := 3
		parallelThreshold := 100
		probeMargin := 0.15
		return rpt.NewRPTIndex(dimension, leafCapacity, candidateProjections, parallelThreshold,
			probeMargin, core.Distances[distanceName], distanceName)
	}

	example.RunDataset(factory, "deep-image-96-angular",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}

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

	// Using RPT index with FashionMNIST and SIFT datasets
	RPTIndexFashionMNIST()
	RPTIndexSIFT()
}

func RPTIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		leafCapacity := 10
		candidateProjections := 3
		parallelThreshold := 100
		probeMargin := 0.15
		return rpt.NewRPTIndex(dimension, leafCapacity, candidateProjections, parallelThreshold,
			probeMargin)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func RPTIndexSIFT() {
	factory := func() core.Index {
		dimension := 128
		leafCapacity := 10
		candidateProjections := 3
		parallelThreshold := 100
		probeMargin := 0.15
		return rpt.NewRPTIndex(dimension, leafCapacity, candidateProjections, parallelThreshold,
			probeMargin)
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

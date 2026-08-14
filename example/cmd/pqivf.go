//go:build ignore
// +build ignore

package main

import (
	"log"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/pqivf"
)

func main() {
	// Using PQIVF index with FashionMNIST and SIFT datasets
	PQIVFIndexFashionMNIST()
	PQIVFIndexSIFT()
}

func PQIVFIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		index, err := pqivf.New(dimension, pqivf.WithPQK(256))
		if err != nil {
			log.Fatalf("Failed to create PQIVF index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func PQIVFIndexSIFT() {
	factory := func() core.Index {
		dimension := 128
		index, err := pqivf.New(dimension, pqivf.WithPQK(256), pqivf.WithKMeansIters(25))
		if err != nil {
			log.Fatalf("Failed to create PQIVF index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

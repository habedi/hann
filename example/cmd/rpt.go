//go:build ignore
// +build ignore

package main

import (
	"log"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/rpt"
)

func main() {
	// Using RPT index with FashionMNIST and SIFT datasets
	RPTIndexFashionMNIST()
	RPTIndexSIFT()
}

func RPTIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		index, err := rpt.New(dimension)
		if err != nil {
			log.Fatalf("Failed to create RPT index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func RPTIndexSIFT() {
	factory := func() core.Index {
		dimension := 128
		index, err := rpt.New(dimension)
		if err != nil {
			log.Fatalf("Failed to create RPT index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

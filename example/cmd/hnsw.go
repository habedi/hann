//go:build ignore
// +build ignore

package main

import (
	"log"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/hnsw"
)

func main() {
	// Run the HNSW index on the FashionMNIST and Glove datasets.
	HNSWIndexFashionMNIST()
	HNSWIndexGlove25()
	HNSWIndexGlove200()
}

func HNSWIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		index, err := hnsw.New(dimension)
		if err != nil {
			log.Fatalf("Failed to create HNSW index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func HNSWIndexGlove25() {
	factory := func() core.Index {
		dimension := 25
		index, err := hnsw.New(dimension, hnsw.WithMetric(core.Cosine))
		if err != nil {
			log.Fatalf("Failed to create HNSW index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "glove-25-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

func HNSWIndexGlove200() {
	factory := func() core.Index {
		dimension := 200
		index, err := hnsw.New(dimension, hnsw.WithMetric(core.Cosine))
		if err != nil {
			log.Fatalf("Failed to create HNSW index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "glove-200-angular",
		"example/data/nearest-neighbors-datasets", 100, 5, 5)
}

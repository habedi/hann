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
	// Using HNSW index with GIST and DEEP1B datasets
	HNSWIndexGIST("euclidean")
	HNSWIndexDEEP1B("cosine")
}

func HNSWIndexGIST(distanceName string) {
	factory := func() core.Index {
		dimension := 960
		metric, ok := core.MetricByName(distanceName)
		if !ok {
			log.Fatalf("Unknown metric: %s", distanceName)
		}
		index, err := hnsw.New(dimension, hnsw.WithMetric(metric))
		if err != nil {
			log.Fatalf("Failed to create HNSW index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "gist-960-euclidean",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}

func HNSWIndexDEEP1B(distanceName string) {
	factory := func() core.Index {
		dimension := 96
		metric, ok := core.MetricByName(distanceName)
		if !ok {
			log.Fatalf("Unknown metric: %s", distanceName)
		}
		index, err := hnsw.New(dimension, hnsw.WithMetric(metric))
		if err != nil {
			log.Fatalf("Failed to create HNSW index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "deep-image-96-angular",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}

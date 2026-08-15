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
	// Run the PQIVF index on the GIST dataset.
	PQIVFIndexGIST()
}

func PQIVFIndexGIST() {
	factory := func() core.Index {
		dimension := 960
		index, err := pqivf.New(dimension, pqivf.WithPQK(256))
		if err != nil {
			log.Fatalf("Failed to create PQIVF index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "gist-960-euclidean",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}

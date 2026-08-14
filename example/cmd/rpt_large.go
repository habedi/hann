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
	// Using RPT index with GIST dataset
	RPTIndexGIST()
}

func RPTIndexGIST() {
	factory := func() core.Index {
		dimension := 960
		index, err := rpt.New(dimension)
		if err != nil {
			log.Fatalf("Failed to create RPT index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "gist-960-euclidean",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}

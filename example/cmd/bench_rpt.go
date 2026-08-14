//go:build ignore
// +build ignore

package main

import (
	"log"
	"net/http"
	_ "net/http/pprof"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/rpt"
)

func main() {
	// Start the pprof HTTP server on port 6060.
	// This will expose profiling endpoints at /debug/pprof/
	go func() {
		log.Println("Starting pprof server on :6060")
		if err := http.ListenAndServe("localhost:6060", nil); err != nil {
			log.Printf("pprof server failed: %v", err)
		}
	}()

	// Benchmarking RPT index with FashionMNIST and SIFT datasets
	BenchRPTIndexFashionMNIST()
	BenchRPTIndexSIFT()
}

func BenchRPTIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		index, err := rpt.New(dimension)
		if err != nil {
			log.Fatalf("Failed to create RPT index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

func BenchRPTIndexSIFT() {
	factory := func() core.Index {
		dimension := 128
		index, err := rpt.New(dimension)
		if err != nil {
			log.Fatalf("Failed to create RPT index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

//go:build ignore
// +build ignore

package main

import (
	"log"
	"net/http"
	_ "net/http/pprof"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/example"
	"github.com/habedi/hann/pqivf"
)

func main() {
	// Start the pprof HTTP server on port 6060. It exposes the profiling
	// endpoints at /debug/pprof/.
	go func() {
		log.Println("Starting pprof server on :6060")
		if err := http.ListenAndServe("localhost:6060", nil); err != nil {
			log.Printf("pprof server failed: %v", err)
		}
	}()

	// Benchmark the PQIVF index on the FashionMNIST and SIFT datasets.
	BenchPQIVFIndexFashionMNIST()
	BenchPQIVFIndexSIFT()
}

func BenchPQIVFIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		index, err := pqivf.New(dimension, pqivf.WithPQK(256))
		if err != nil {
			log.Fatalf("Failed to create PQIVF index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

func BenchPQIVFIndexSIFT() {
	factory := func() core.Index {
		dimension := 128
		index, err := pqivf.New(dimension, pqivf.WithPQK(256), pqivf.WithKMeansIters(25))
		if err != nil {
			log.Fatalf("Failed to create PQIVF index: %v", err)
		}
		return index
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

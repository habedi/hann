//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"os"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/hnsw"
)

// Note: results may vary between different runs even if HANN_SEED is set.
// That's expected as the HNSW index uses none-deterministic operations (like parallel loops).

func main() {

	// Set the logger to output to the console.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Index parameters.
	dim := 6
	m := 5
	ef := 10
	distanceName := "euclidean"

	// Create an HNSW index with the given parameters.
	index := hnsw.NewHNSW(dim, m, ef, core.Distances[distanceName], distanceName)
	fmt.Println("Created new HNSW index.")

	// Add a few vectors.
	fmt.Println("Adding vectors...")
	vectors := map[int][]float32{
		1:  {1, 2, 3, 4, 5, 6},
		2:  {6, 5, 4, 3, 2, 1},
		3:  {1, 1, 1, 1, 1, 1},
		4:  {2, 2, 2, 2, 2, 2},
		5:  {3, 3, 3, 3, 3, 3},
		6:  {4, 4, 4, 4, 4, 4},
		7:  {5, 5, 5, 5, 5, 5},
		8:  {6, 6, 6, 6, 6, 6},
		9:  {7, 7, 7, 7, 7, 7},
		10: {8, 8, 8, 8, 8, 8},
	}
	for id, vec := range vectors {
		if err := index.Add(id, vec); err != nil {
			log.Fatal().Msgf("Add failed for id %d: %v", id, err)
		}
	}
	fmt.Printf("Index stats after Add: %+v\n", index.Stats())

	// Search for the nearest neighbors of a query vector.
	query := []float32{1, 2, 3, 4, 5, 6}
	fmt.Println("Searching nearest neighbors for vector:", query)
	neighbors, err := index.Search(query, 2)
	if err != nil {
		log.Fatal().Msgf("Search failed: %v", err)
	}
	fmt.Println("Search results:")
	for _, n := range neighbors {
		fmt.Printf("ID: %d, Distance: %f\n", n.ID, n.Distance)
	}

	// Update a vector.
	fmt.Println("Updating vector with id 2...")
	newVec := []float32{2, 2, 2, 2, 2, 2}
	if err := index.Update(2, newVec); err != nil {
		log.Fatal().Msgf("Update failed: %v", err)
	}
	fmt.Printf("Index stats after Update: %+v\n", index.Stats())

	// Delete a vector.
	fmt.Println("Deleting vector with id 3...")
	if err := index.Delete(3); err != nil {
		log.Fatal().Msgf("Delete failed: %v", err)
	}
	fmt.Printf("Index stats after Delete: %+v\n", index.Stats())

	// Save the index to disk.
	filePath := "hnsw_index.gob"
	fmt.Println("Saving index to file:", filePath)
	if err := index.Save(filePath); err != nil {
		log.Fatal().Msgf("Save failed: %v", err)
	}

	// Create a new index instance and load the saved index into it.
	fmt.Println("Loading index from file:", filePath)
	newIndex := hnsw.NewHNSW(dim, m, ef, core.Distances[distanceName], distanceName)
	if err := newIndex.Load(filePath); err != nil {
		log.Fatal().Msgf("Load failed: %v", err)
	}
	fmt.Printf("Index stats after Load: %+v\n", newIndex.Stats())

	// Search in the loaded index.
	fmt.Println("Searching in loaded index...")
	neighbors, err = newIndex.Search(query, 2)
	if err != nil {
		log.Fatal().Msgf("Search in loaded index failed: %v", err)
	}
	fmt.Println("Search results from loaded index:")
	for _, n := range neighbors {
		fmt.Printf("ID: %d, Distance: %f\n", n.ID, n.Distance)
	}

	// Remove the index file now that we don't need it anymore.
	if err := os.Remove(filePath); err != nil {
		log.Printf("Warning: could not remove temporary file %s: %v", filePath, err)
	}
}

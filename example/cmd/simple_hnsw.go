//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"log"
	"os"

	"github.com/habedi/hann/hnsw"
)

// Note: results may vary between runs even if HANN_SEED is set. That is
// expected, because the HNSW index uses non-deterministic operations, such
// as parallel loops.

func main() {

	// Index parameters.
	dim := 6
	m := 5
	ef := 10

	// Create an HNSW index with the given parameters.
	index, err := hnsw.New(dim, hnsw.WithM(m), hnsw.WithEf(ef))
	if err != nil {
		log.Fatalf("Failed to create HNSW index: %v", err)
	}
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
			log.Fatalf("Add failed for id %d: %v", id, err)
		}
	}
	fmt.Printf("Index stats after Add: %+v\n", index.Stats())

	// Search for the nearest neighbors of a query vector.
	query := []float32{1, 2, 3, 4, 5, 6}
	fmt.Println("Searching nearest neighbors for vector:", query)
	neighbors, err := index.Search(query, 2)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	fmt.Println("Search results:")
	for _, n := range neighbors {
		fmt.Printf("ID: %d, Distance: %f\n", n.ID, n.Distance)
	}

	// Update a vector.
	fmt.Println("Updating vector with id 2...")
	newVec := []float32{2, 2, 2, 2, 2, 2}
	if err := index.Update(2, newVec); err != nil {
		log.Fatalf("Update failed: %v", err)
	}
	fmt.Printf("Index stats after Update: %+v\n", index.Stats())

	// Delete a vector.
	fmt.Println("Deleting vector with id 3...")
	if err := index.Delete(3); err != nil {
		log.Fatalf("Delete failed: %v", err)
	}
	fmt.Printf("Index stats after Delete: %+v\n", index.Stats())

	// Save the index to disk.
	filePath := "hnsw_index.gob"
	fmt.Println("Saving index to file:", filePath)
	saveFile, err := os.Create(filePath)
	if err != nil {
		log.Fatalf("failed to create file: %v", err)
	}
	if err := index.Save(saveFile); err != nil {
		log.Fatalf("Save failed: %v", err)
	}
	saveFile.Close()

	// Create a new index and load the saved state from the io.Reader.
	fmt.Println("Loading index from file:", filePath)
	loadFile, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("failed to open file: %v", err)
	}
	newIndex, err := hnsw.New(dim, hnsw.WithM(m), hnsw.WithEf(ef))
	if err != nil {
		log.Fatalf("Failed to create HNSW index: %v", err)
	}
	if err := newIndex.Load(loadFile); err != nil {
		log.Fatalf("Load failed: %v", err)
	}
	loadFile.Close()
	fmt.Printf("Index stats after Load: %+v\n", newIndex.Stats())

	// Search in the loaded index.
	fmt.Println("Searching in loaded index...")
	neighbors, err = newIndex.Search(query, 2)
	if err != nil {
		log.Fatalf("Search in loaded index failed: %v", err)
	}
	fmt.Println("Search results from loaded index:")
	for _, n := range neighbors {
		fmt.Printf("ID: %d, Distance: %f\n", n.ID, n.Distance)
	}

	// Remove the index file. It is no longer needed.
	if err := os.Remove(filePath); err != nil {
		log.Printf("Warning: could not remove temporary file %s: %v", filePath, err)
	}
}

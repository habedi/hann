package example

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/habedi/hann/core"
	"github.com/rs/zerolog/log"
)

// LoadDataset loads a dataset from a directory into the given index.
// The directory must contain the following files:
//   - train.csv       (vectors to add to the index)
//   - test.csv        (query vectors, not added to the index)
//   - neighbors.csv   (expected neighbor IDs per query)
//   - distances.csv   (expected distances per query)
func LoadDataset(index core.Index, dir string) (
	testVectors [][]float32,
	trueNeighbors [][]int,
	trueDistances [][]float64,
	err error,
) {
	log.Info().Msgf("Loading dataset from directory: %s", dir)

	trainPath := filepath.Join(dir, "train.csv")
	testPath := filepath.Join(dir, "test.csv")
	neighborsPath := filepath.Join(dir, "neighbors.csv")
	distancesPath := filepath.Join(dir, "distances.csv")

	// Load training vectors into the index.
	log.Info().Msgf("Loading training data from: %s", trainPath)
	if err := LoadCSV(index, trainPath, false); err != nil {
		return nil, nil, nil,
			fmt.Errorf("failed to load train.csv: %w", err)
	}

	// Load test vectors (not added to the index).
	log.Info().Msgf("Loading test data from: %s", testPath)
	testVectors, err = readCSV[float32](testPath, false)
	if err != nil {
		return nil, nil, nil,
			fmt.Errorf("failed to load test.csv: %w", err)
	}

	// Load ground-truth neighbors.
	log.Info().Msgf("Loading ground-truth neighbors from: %s", neighborsPath)
	trueNeighbors, err = readCSV[int](neighborsPath, false)
	if err != nil {
		return nil, nil, nil,
			fmt.Errorf("failed to load neighbors.csv: %w", err)
	}

	// Load ground-truth distances.
	log.Info().Msgf("Loading ground-truth distances from: %s", distancesPath)
	trueDistances, err = readCSV[float64](distancesPath, false)
	if err != nil {
		return nil, nil, nil,
			fmt.Errorf("failed to load distances.csv: %w", err)
	}

	log.Info().Msg("Dataset loaded successfully")
	return testVectors, trueNeighbors, trueDistances, nil
}

// LoadCSV reads float32 vectors from a CSV file and adds them to the index.
func LoadCSV(index core.Index, path string, skipHeader bool) error {
	log.Info().Msgf("Loading CSV file into index: %s", path)
	vectors, err := readCSV[float32](path, skipHeader)
	if err != nil {
		return err
	}
	// Do not adjust IDs. Use 0-indexing to match ground-truth.
	for id, vec := range vectors {
		if err := index.Add(id, vec); err != nil {
			return fmt.Errorf("failed to add vector %d: %w", id, err)
		}
	}
	log.Info().Msgf("Loaded %d vectors from %s", len(vectors), path)
	return nil
}

// readCSV is a generic CSV reader for types: int, float32, and float64.
func readCSV[T int | float32 | float64](path string, skipHeader bool) ([][]T, error) {
	log.Debug().Msgf("Opening CSV file: %s", path)
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var result [][]T

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read error in %s: %w", path, err)
		}
		if skipHeader {
			skipHeader = false
			continue
		}
		row := make([]T, len(record))
		for i, val := range record {
			parsed, err := parseValue[T](val)
			if err != nil {
				return nil, fmt.Errorf("parse error at col %d in %s: %w", i, path, err)
			}
			row[i] = parsed
		}
		result = append(result, row)
	}

	log.Debug().Msgf("Parsed %d rows from %s", len(result), path)
	return result, nil
}

// parseValue converts a string to T (int, float32, or float64).
func parseValue[T int | float32 | float64](s string) (T, error) {
	s = strings.TrimSpace(s)
	var zero T
	switch any(zero).(type) {
	case int:
		v, err := strconv.Atoi(s)
		return any(v).(T), err
	case float32:
		v, err := strconv.ParseFloat(s, 32)
		return any(float32(v)).(T), err
	case float64:
		v, err := strconv.ParseFloat(s, 64)
		return any(v).(T), err
	default:
		return zero, fmt.Errorf("unsupported type %T", zero)
	}
}

// LoadTrainingVectors loads training vectors from "train.csv" in the specified directory.
// It returns a map from id (row number, 0-indexed) to the vector.
func LoadTrainingVectors(dir string) (map[int][]float32, error) {
	trainPath := filepath.Join(dir, "train.csv")
	log.Info().Msgf("Loading training vectors from: %s", trainPath)
	// reuse generic CSV reader (no header in these CSV files)
	vectors, err := readCSV[float32](trainPath, false)
	if err != nil {
		return nil, fmt.Errorf("failed to load train.csv: %w", err)
	}
	m := make(map[int][]float32, len(vectors))
	for id, vec := range vectors {
		m[id] = vec
	}
	log.Info().Msgf("Loaded %d training vectors from %s", len(m), trainPath)
	return m, nil
}

// LoadTestDataset loads the test vectors and ground-truth data from the specified directory.
// It returns the test vectors, true neighbor IDs, and true distances (ground-truth).
func LoadTestDataset(dir string) ([][]float32, [][]int, [][]float64, error) {
	testPath := filepath.Join(dir, "test.csv")
	neighborsPath := filepath.Join(dir, "neighbors.csv")
	distancesPath := filepath.Join(dir, "distances.csv")

	log.Info().Msgf("Loading test vectors from: %s", testPath)
	testVectors, err := readCSV[float32](testPath, false)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load test.csv: %w", err)
	}

	log.Info().Msgf("Loading ground-truth neighbors from: %s", neighborsPath)
	trueNeighbors, err := readCSV[int](neighborsPath, false)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load neighbors.csv: %w", err)
	}

	log.Info().Msgf("Loading ground-truth distances from: %s", distancesPath)
	trueDistances, err := readCSV[float64](distancesPath, false)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load distances.csv: %w", err)
	}

	return testVectors, trueNeighbors, trueDistances, nil
}

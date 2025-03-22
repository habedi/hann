package examples

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/habedi/hann/core"
)

// LoadDataset loads a dataset from a directory into the given index.
func LoadDataset(index core.Index, dir string) (
	testVectors [][]float32,
	trueNeighbors [][]int,
	trueDistances [][]float64,
	err error,
) {
	if err := LoadCSV(index, filepath.Join(dir, "train.csv"), true); err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load train.csv: %w", err)
	}

	testVectors, err = readCSV[float32](filepath.Join(dir, "test.csv"), true)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load test.csv: %w", err)
	}

	trueNeighbors, err = readCSV[int](filepath.Join(dir, "neighbors.csv"), true)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load neighbors.csv: %w", err)
	}

	trueDistances, err = readCSV[float64](filepath.Join(dir, "distances.csv"), true)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load distances.csv: %w", err)
	}

	return testVectors, trueNeighbors, trueDistances, nil
}

// LoadCSV reads float32 vectors from a CSV and adds them to the index.
func LoadCSV(index core.Index, path string, skipHeader bool) error {
	vectors, err := readCSV[float32](path, skipHeader)
	if err != nil {
		return err
	}
	for id, vec := range vectors {
		if err := index.Add(id, vec); err != nil {
			return fmt.Errorf("failed to add vector %d: %w", id, err)
		}
	}
	return nil
}

// readCSV is a generic CSV reader for types: int, float32, float64.
func readCSV[T int | float32 | float64](path string, skipHeader bool) ([][]T, error) {
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

	return result, nil
}

// parseValue converts a string to T: int, float32, or float64.
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

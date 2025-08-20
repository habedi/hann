package core

import (
	"math"
	"testing"
)

func TestNormalizeVector(t *testing.T) {
	tests := []struct {
		vec      []float32
		expected []float32
	}{
		{
			vec: []float32{1, 1, 1, 1, 1, 1, 1, 1},
			expected: []float32{0.353553, 0.353553, 0.353553, 0.353553,
				0.353553, 0.353553, 0.353553, 0.353553},
		},
		{
			vec:      []float32{8, 0, 0, 0, 0, 0, 0, 0},
			expected: []float32{1, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			vec:      []float32{0, 0, 0, 0, 0, 0, 0, 0},
			expected: []float32{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			vec:      []float32{},
			expected: []float32{},
		},
	}

	for _, tt := range tests {
		NormalizeVector(tt.vec)

		for i := range tt.vec {
			if math.Abs(float64(tt.vec[i]-tt.expected[i])) > 1e-5 {
				t.Errorf("NormalizeVector failed.\nGot:      %v\nExpected: %v",
					tt.vec, tt.expected)
				break
			}
		}
	}
}

func TestNormalizeBatchEdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		vecs     [][]float32
		expected [][]float32
	}{
		{
			name:     "Empty batch",
			vecs:     [][]float32{},
			expected: [][]float32{},
		},
		{
			name:     "Batch with empty vector",
			vecs:     [][]float32{{1, 2, 3}, {}, {4, 5, 6}},
			expected: [][]float32{{0.267261, 0.534522, 0.801784}, {}, {0.455842, 0.569803, 0.683763}},
		},
		{
			name:     "Batch with different length vectors",
			vecs:     [][]float32{{1, 2}, {3, 4, 5}},
			expected: [][]float32{{0.447214, 0.894427}, {0.424264, 0.565685, 0.707107}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			NormalizeBatch(tt.vecs)
			if len(tt.vecs) == 0 && len(tt.expected) == 0 {
				return // success
			}

			for idx, vec := range tt.vecs {
				if len(vec) == 0 && len(tt.expected[idx]) == 0 {
					continue // success
				}
				for i := range vec {
					if math.Abs(float64(vec[i]-tt.expected[idx][i])) > 1e-5 {
						t.Errorf("NormalizeBatch edge cases failed at vector %d.\nGot:      %v\nExpected: %v",
							idx, vec, tt.expected[idx])
						break
					}
				}
			}
		})
	}
}

func TestNormalizeBatch(t *testing.T) {
	vecs := [][]float32{
		{3, 0, 4, 0, 0, 0, 0, 0},
		{1, 2, 2, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
	}

	expected := [][]float32{
		{0.6, 0, 0.8, 0, 0, 0, 0, 0},
		{0.333333, 0.666666, 0.666666, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
	}

	NormalizeBatch(vecs)

	for idx, vec := range vecs {
		for i := range vec {
			if math.Abs(float64(vec[i]-expected[idx][i])) > 1e-5 {
				t.Errorf("NormalizeBatch failed at vector %d.\nGot:      %v\nExpected: %v",
					idx, vec, expected[idx])
				break
			}
		}
	}
}

func TestNormalizeBatchLarge(t *testing.T) {
	// Generate a batch of 100 vectors, each with 8 dimensions
	numVecs := 100
	vecLen := 8
	vecs := make([][]float32, numVecs)

	// Initialize vectors with incremental values
	for i := 0; i < numVecs; i++ {
		vec := make([]float32, vecLen)
		for j := 0; j < vecLen; j++ {
			vec[j] = float32(j + 1)
		}
		vecs[i] = vec
	}

	// Expected normalized vector (since all are identical)
	norm := float32(math.Sqrt(204)) // sqrt(1²+2²+3²+4²+5²+6²+7²+8²) = sqrt(204)
	expected := []float32{
		1 / norm, 2 / norm, 3 / norm, 4 / norm,
		5 / norm, 6 / norm, 7 / norm, 8 / norm,
	}

	NormalizeBatch(vecs)

	// Check each normalized vector
	for idx, vec := range vecs {
		for i := range vec {
			if math.Abs(float64(vec[i]-expected[i])) > 1e-5 {
				t.Errorf("NormalizeBatchLarge failed at vector %d.\nGot:      %v\nExpected: %v",
					idx, vec, expected)
				break
			}
		}
	}
}

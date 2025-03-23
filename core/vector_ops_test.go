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
	}

	for _, tt := range tests {
		NormalizeVector(tt.vec)

		for i := range tt.vec {
			if math.Abs(float64(tt.vec[i]-tt.expected[i])) > 1e-5 {
				t.Errorf("NormalizeVector failed.\nGot:      %v\nExpected: %v", tt.vec, tt.expected)
				break
			}
		}
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
				t.Errorf("NormalizeBatch failed at vector %d.\nGot:      %v\nExpected: %v", idx, vec, expected[idx])
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
				t.Errorf("NormalizeBatchLarge failed at vector %d.\nGot:      %v\nExpected: %v", idx, vec, expected)
				break
			}
		}
	}
}

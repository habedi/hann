package core

import (
	"math"
	"testing"
)

// almostEqual compares two floating-point values with a tolerance.
func almostEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestDistanceFunctions(t *testing.T) {
	tests := []struct {
		name                     string
		a, b                     []float32
		expectedEuclidean        float64
		expectedSquaredEuclidean float64
		expectedManhattan        float64
		expectedCosineDistance   float64
	}{
		{
			name:                     "Identical Vectors",
			a:                        []float32{1, 2, 3, 4, 5, 6},
			b:                        []float32{1, 2, 3, 4, 5, 6},
			expectedEuclidean:        0,
			expectedSquaredEuclidean: 0,
			expectedManhattan:        0,
			expectedCosineDistance:   0,
		},
		{
			name: "Opposite Order",
			a:    []float32{1, 2, 3, 4, 5, 6},
			b:    []float32{6, 5, 4, 3, 2, 1},

			// Euclidean: sqrt((5^2 + 3^2 + 1^2 + 1^2 + 3^2 + 5^2)) = sqrt(70), squared = 70, Manhattan = 18.
			expectedEuclidean:        math.Sqrt(70),
			expectedSquaredEuclidean: 70,
			expectedManhattan:        18,

			// Cosine: similarity = 56 / 91, so cosine distance = 1 - (56/91).
			expectedCosineDistance: 1 - (56.0 / 91.0),
		},
		{
			name: "Binary Opposites",
			a:    []float32{1, 0, 0, 1, 0, 1},
			b:    []float32{0, 1, 1, 0, 1, 0},

			// Euclidean: sqrt(1+1+1+1+1+1) = sqrt(6), squared = 6, Manhattan = 6.
			expectedEuclidean:        math.Sqrt(6),
			expectedSquaredEuclidean: 6,
			expectedManhattan:        6,

			// Cosine: dot = 0, so cosine similarity is 0 and cosine distance = 1.
			expectedCosineDistance: 1,
		},
		{
			name: "Negative Numbers",
			a:    []float32{-1, -2, -3, -4, -5, -6},
			b:    []float32{-6, -5, -4, -3, -2, -1},

			// Euclidean: sqrt((5^2 + 3^2 + 1^2 + 1^2 + 3^2 + 5^2)) = sqrt(70), squared = 70, Manhattan = 18.
			expectedEuclidean:        math.Sqrt(70),
			expectedSquaredEuclidean: 70,
			expectedManhattan:        18,

			// Cosine: similarity = 56 / 91, so cosine distance = 1 - (56/91).
			expectedCosineDistance: 1 - (56.0 / 91.0),
		},
		{
			name:                     "Zero Vector Cosine",
			a:                        []float32{0, 0, 0, 0, 0, 0},
			b:                        []float32{1, 2, 3, 4, 5, 6},
			expectedEuclidean:        math.Sqrt(91),
			expectedSquaredEuclidean: 91,
			expectedManhattan:        21,
			expectedCosineDistance:   1, // Special case for zero vector
		},
		{
			name: "Near Identical",
			a:    []float32{1, 2, 3, 4, 5, 6},
			b:    []float32{1, 2, 3, 4, 5, 6.000001},
			// Expect cosine distance to be very close to 0, but not exactly 0.
			// The clamping logic will ensure it's not negative.
			expectedEuclidean:        1e-6,
			expectedSquaredEuclidean: 1e-12,
			expectedManhattan:        1e-6,
			expectedCosineDistance:   0, // due to clamping and float precision
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: set up test data (see above).
			// Act: compute distances using the core package functions.
			euclid := Euclidean(tt.a, tt.b)
			sqEuclid := SquaredEuclidean(tt.a, tt.b)
			manhattan := Manhattan(tt.a, tt.b)
			cosine := CosineDistance(tt.a, tt.b)

			// Assert: compare computed values with expected ones.
			if !almostEqual(euclid, tt.expectedEuclidean, 1e-6) {
				t.Errorf("Euclidean(%v, %v) = %v; want %v", tt.a, tt.b, euclid,
					tt.expectedEuclidean)
			}
			if !almostEqual(sqEuclid, tt.expectedSquaredEuclidean, 1e-6) {
				t.Errorf("SquaredEuclidean(%v, %v) = %v; want %v", tt.a, tt.b, sqEuclid,
					tt.expectedSquaredEuclidean)
			}
			if !almostEqual(manhattan, tt.expectedManhattan, 1e-6) {
				t.Errorf("Manhattan(%v, %v) = %v; want %v", tt.a, tt.b, manhattan,
					tt.expectedManhattan)
			}
			if !almostEqual(cosine, tt.expectedCosineDistance, 1e-6) {
				t.Errorf("CosineDistance(%v, %v) = %v; want %v", tt.a, tt.b, cosine,
					tt.expectedCosineDistance)
			}
		})
	}
}

func TestDistanceFunctionsPanics(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		fn   func(a, b []float32) float64
	}{
		{"Euclidean different length", []float32{1}, []float32{1, 2}, Euclidean},
		{"Euclidean empty a", []float32{}, []float32{1, 2}, Euclidean},
		{"Euclidean empty b", []float32{1, 2}, []float32{}, Euclidean},
		{"SquaredEuclidean different length", []float32{1}, []float32{1, 2}, SquaredEuclidean},
		{"SquaredEuclidean empty a", []float32{}, []float32{1, 2}, SquaredEuclidean},
		{"SquaredEuclidean empty b", []float32{1, 2}, []float32{}, SquaredEuclidean},
		{"Manhattan different length", []float32{1}, []float32{1, 2}, Manhattan},
		{"Manhattan empty a", []float32{}, []float32{1, 2}, Manhattan},
		{"Manhattan empty b", []float32{1, 2}, []float32{}, Manhattan},
		{"CosineDistance different length", []float32{1}, []float32{1, 2}, CosineDistance},
		{"CosineDistance empty a", []float32{}, []float32{1, 2}, CosineDistance},
		{"CosineDistance empty b", []float32{1, 2}, []float32{}, CosineDistance},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("The code did not panic")
				}
			}()
			tt.fn(tt.a, tt.b)
		})
	}
}

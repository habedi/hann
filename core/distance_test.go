package core

import (
	"math"
	"math/rand"
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
			name:                     "Opposite Order",
			a:                        []float32{1, 2, 3, 4, 5, 6},
			b:                        []float32{6, 5, 4, 3, 2, 1},
			expectedEuclidean:        math.Sqrt(70),
			expectedSquaredEuclidean: 70,
			expectedManhattan:        18,
			expectedCosineDistance:   1 - (56.0 / 91.0),
		},
		{
			name:                     "Binary Opposites",
			a:                        []float32{1, 0, 0, 1, 0, 1},
			b:                        []float32{0, 1, 1, 0, 1, 0},
			expectedEuclidean:        math.Sqrt(6),
			expectedSquaredEuclidean: 6,
			expectedManhattan:        6,
			expectedCosineDistance:   1,
		},
		{
			name:                     "Negative Numbers",
			a:                        []float32{-1, -2, -3, -4, -5, -6},
			b:                        []float32{-6, -5, -4, -3, -2, -1},
			expectedEuclidean:        math.Sqrt(70),
			expectedSquaredEuclidean: 70,
			expectedManhattan:        18,
			expectedCosineDistance:   1 - (56.0 / 91.0),
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
			name:                     "Near Identical",
			a:                        []float32{1, 2, 3, 4, 5, 6},
			b:                        []float32{1, 2, 3, 4, 5, 6.000001},
			expectedEuclidean:        1e-6,
			expectedSquaredEuclidean: 1e-12,
			expectedManhattan:        1e-6,
			expectedCosineDistance:   0, // due to clamping and float precision
		},
		{
			name:                     "Empty Vectors",
			a:                        []float32{},
			b:                        []float32{},
			expectedEuclidean:        0,
			expectedSquaredEuclidean: 0,
			expectedManhattan:        0,
			expectedCosineDistance:   0,
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			euclid, err := Euclidean(tt.a, tt.b)
			if err != nil {
				t.Errorf("Euclidean returned an unexpected error: %v", err)
			}
			if !almostEqual(euclid, tt.expectedEuclidean, 1e-6) {
				t.Errorf("Euclidean(%v, %v) = %v; want %v", tt.a, tt.b, euclid, tt.expectedEuclidean)
			}

			sqEuclid, err := SquaredEuclidean(tt.a, tt.b)
			if err != nil {
				t.Errorf("SquaredEuclidean returned an unexpected error: %v", err)
			}
			if !almostEqual(sqEuclid, tt.expectedSquaredEuclidean, 1e-6) {
				t.Errorf("SquaredEuclidean(%v, %v) = %v; want %v", tt.a, tt.b, sqEuclid, tt.expectedSquaredEuclidean)
			}

			manhattan, err := Manhattan(tt.a, tt.b)
			if err != nil {
				t.Errorf("Manhattan returned an unexpected error: %v", err)
			}
			if !almostEqual(manhattan, tt.expectedManhattan, 1e-6) {
				t.Errorf("Manhattan(%v, %v) = %v; want %v", tt.a, tt.b, manhattan, tt.expectedManhattan)
			}

			cosine, err := CosineDistance(tt.a, tt.b)
			if err != nil {
				t.Errorf("CosineDistance returned an unexpected error: %v", err)
			}
			if !almostEqual(cosine, tt.expectedCosineDistance, 1e-6) {
				t.Errorf("CosineDistance(%v, %v) = %v; want %v", tt.a, tt.b, cosine, tt.expectedCosineDistance)
			}
		})
	}
}

func TestDistanceFunctionsErrors(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		fn   func(a, b []float32) (float64, error)
	}{
		{"Euclidean different length", []float32{1}, []float32{1, 2}, Euclidean},
		{"SquaredEuclidean different length", []float32{1}, []float32{1, 2}, SquaredEuclidean},
		{"Manhattan different length", []float32{1}, []float32{1, 2}, Manhattan},
		{"CosineDistance different length", []float32{1}, []float32{1, 2}, CosineDistance},
		{"Euclidean empty a", []float32{}, []float32{1, 2}, Euclidean},
		{"SquaredEuclidean empty a", []float32{}, []float32{1, 2}, SquaredEuclidean},
		{"Manhattan empty a", []float32{}, []float32{1, 2}, Manhattan},
		{"CosineDistance empty a", []float32{}, []float32{1, 2}, CosineDistance},
		{"Euclidean empty b", []float32{1, 2}, []float32{}, Euclidean},
		{"SquaredEuclidean empty b", []float32{1, 2}, []float32{}, SquaredEuclidean},
		{"Manhattan empty b", []float32{1, 2}, []float32{}, Manhattan},
		{"CosineDistance empty b", []float32{1, 2}, []float32{}, CosineDistance},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.fn(tt.a, tt.b)
			if err == nil {
				t.Errorf("Expected an error but got nil")
			}
		})
	}
}

// referenceDistance computes a distance in float64 arithmetic, independently
// of the C implementations, for differential testing. The zero-norm and
// clamping behavior mirrors the documented behavior of the cosine kernel.
func referenceDistance(name string, a, b []float32) float64 {
	switch name {
	case "euclidean":
		return math.Sqrt(referenceDistance("squared_euclidean", a, b))
	case "squared_euclidean":
		sum := 0.0
		for i := range a {
			d := float64(a[i]) - float64(b[i])
			sum += d * d
		}
		return sum
	case "manhattan":
		sum := 0.0
		for i := range a {
			sum += math.Abs(float64(a[i]) - float64(b[i]))
		}
		return sum
	case "cosine":
		dot, normA, normB := 0.0, 0.0, 0.0
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
			normA += float64(a[i]) * float64(a[i])
			normB += float64(b[i]) * float64(b[i])
		}
		if normA == 0 || normB == 0 {
			return 1.0
		}
		sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
		if sim > 1 {
			sim = 1
		}
		if sim < -1 {
			sim = -1
		}
		return 1.0 - sim
	}
	panic("unknown distance name " + name)
}

// TestDistanceDifferentialAgainstReference compares every distance function,
// through whichever SIMD variant is installed on this machine, against an
// independent float64 reference over random vectors. The dimensions sweep the
// vector-lane boundaries so the tail loops of the SIMD variants are covered.
func TestDistanceDifferentialAgainstReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	dims := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 25, 31, 32, 33, 128, 200, 784}
	for name, fn := range Distances {
		for _, dim := range dims {
			for trial := 0; trial < 5; trial++ {
				a := make([]float32, dim)
				b := make([]float32, dim)
				for i := 0; i < dim; i++ {
					a[i] = rng.Float32()*20 - 10
					b[i] = rng.Float32()*20 - 10
				}
				got, err := fn(a, b)
				if err != nil {
					t.Fatalf("%s dim %d: unexpected error: %v", name, dim, err)
				}
				want := referenceDistance(name, a, b)
				tol := 1e-3 * math.Max(1, math.Abs(want))
				if math.Abs(got-want) > tol {
					t.Errorf("%s dim %d trial %d: got %v, reference %v", name, dim, trial, got, want)
				}
			}
		}
	}

	// The zero-vector operand is the documented cosine edge case.
	zero := make([]float32, 8)
	other := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	got, err := CosineDistance(zero, other)
	if err != nil {
		t.Fatalf("cosine with zero vector: unexpected error: %v", err)
	}
	if got != 1.0 {
		t.Errorf("cosine with zero vector: got %v, want 1.0", got)
	}
}

package core

import (
	"fmt"
	"sync"
)

// Metric bundles a distance function with its name and its normalization
// requirement, so the three always travel together. The name is what Stats
// reports and what the serialized form of an index stores, and the
// normalization flag replaces the old convention of switching behavior on the
// name "cosine".
type Metric struct {
	name      string
	fn        DistanceFunc
	normalize bool
}

// NewMetric returns a metric with the given name, distance function, and
// normalization requirement. Register it with RegisterMetric if indexes using
// it must load from a saved file.
func NewMetric(name string, fn DistanceFunc, normalize bool) Metric {
	return Metric{name: name, fn: fn, normalize: normalize}
}

// Name returns the metric's name.
func (m Metric) Name() string { return m.name }

// Distance computes the distance between two vectors.
func (m Metric) Distance(a, b []float32) (float64, error) { return m.fn(a, b) }

// Func returns the metric's distance function.
func (m Metric) Func() DistanceFunc { return m.fn }

// Normalizes reports whether vectors must be normalized before they are
// stored or searched under this metric.
func (m Metric) Normalizes() bool { return m.normalize }

// IsZero reports whether the metric is the zero value, which carries no
// distance function.
func (m Metric) IsZero() bool { return m.fn == nil }

// The built-in metrics.
var (
	// Euclidean is the Euclidean (L2) distance.
	Euclidean = Metric{name: "euclidean", fn: euclidean}
	// SquaredEuclidean is the squared Euclidean distance.
	SquaredEuclidean = Metric{name: "squared_euclidean", fn: squaredEuclidean}
	// Manhattan is the Manhattan (L1) distance.
	Manhattan = Metric{name: "manhattan", fn: manhattan}
	// Cosine is the cosine distance. Vectors are normalized when stored.
	Cosine = Metric{name: "cosine", fn: cosineDistance, normalize: true}
)

var (
	metricsMu sync.RWMutex
	metrics   = map[string]Metric{
		Euclidean.name:        Euclidean,
		SquaredEuclidean.name: SquaredEuclidean,
		Manhattan.name:        Manhattan,
		Cosine.name:           Cosine,
	}
)

// RegisterMetric adds a custom metric to the registry, so an index saved with
// it can be loaded by name. It returns an error when the metric has no name
// or no distance function, or when the name is already registered.
func RegisterMetric(m Metric) error {
	if m.name == "" {
		return fmt.Errorf("metric has no name")
	}
	if m.fn == nil {
		return fmt.Errorf("metric %q has no distance function", m.name)
	}
	metricsMu.Lock()
	defer metricsMu.Unlock()
	if _, exists := metrics[m.name]; exists {
		return fmt.Errorf("metric %q is already registered", m.name)
	}
	metrics[m.name] = m
	return nil
}

// MetricByName returns the registered metric with the given name.
func MetricByName(name string) (Metric, bool) {
	metricsMu.RLock()
	defer metricsMu.RUnlock()
	m, ok := metrics[name]
	return m, ok
}

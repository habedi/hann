package core

import (
	"errors"
	"fmt"
	"math"
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
	rankFn    DistanceFunc          // order-equivalent, possibly cheaper distance
	fromRank  func(float64) float64 // converts a rank value to the true distance
	batchFn   batchDistanceFunc     // batched form of fn, nil for custom metrics
	rankBatch batchDistanceFunc     // batched form of rankFn, nil where rankFn is nil
}

// batchDistanceFunc computes the distance from one query to every candidate
// vector in a flat buffer, writing one value per candidate into out. The
// caller has already validated the lengths.
type batchDistanceFunc func(query, flat []float32, out []float64)

// NewMetric returns a metric with the given name, distance function, and
// normalization requirement. Register it with RegisterMetric if indexes using
// it must load from a saved file.
func NewMetric(name string, fn DistanceFunc, normalize bool) Metric {
	return Metric{name: name, fn: fn, normalize: normalize}
}

// Rank computes a distance that orders candidates exactly like Distance but
// may be cheaper. For the Euclidean metric it is the squared Euclidean
// distance, which skips a square root per call; for every other metric it is
// Distance itself. Rank values from different metrics are not comparable, and
// a rank value converts to the true distance through FromRank.
func (m Metric) Rank(a, b []float32) (float64, error) {
	if m.rankFn != nil {
		return m.rankFn(a, b)
	}
	return m.fn(a, b)
}

// FromRank converts a value computed by Rank to the true distance.
func (m Metric) FromRank(rank float64) float64 {
	if m.fromRank != nil {
		return m.fromRank(rank)
	}
	return rank
}

// DistanceBatch computes the distance from the query to each of n candidate
// vectors and writes one distance per candidate into out. The candidates are
// stored consecutively in flat, one row of len(query) floats per candidate,
// so n is len(flat) divided by len(query). It returns an error when the query
// is empty, when len(flat) is not a multiple of len(query), or when len(out)
// differs from n; with n equal to zero it is a no-op. A built-in metric
// computes the batch in one C call, and a metric created with NewMetric falls
// back to a loop over Distance.
func (m Metric) DistanceBatch(query, flat []float32, out []float64) error {
	n, err := batchSize(query, flat, out)
	if err != nil || n == 0 {
		return err
	}
	if m.batchFn != nil {
		m.batchFn(query, flat, out)
		return nil
	}
	return batchLoop(m.fn, query, flat, out)
}

// RankBatch computes the rank distance from the query to each of n candidate
// vectors and writes one value per candidate into out. It relates to
// DistanceBatch exactly as Rank relates to Distance: the values order
// candidates the same way but may be cheaper, and they convert to true
// distances through FromRank. The arguments follow the same flat layout and
// the same validation as DistanceBatch.
func (m Metric) RankBatch(query, flat []float32, out []float64) error {
	n, err := batchSize(query, flat, out)
	if err != nil || n == 0 {
		return err
	}
	if m.rankBatch != nil {
		m.rankBatch(query, flat, out)
		return nil
	}
	if m.rankFn == nil && m.batchFn != nil {
		m.batchFn(query, flat, out)
		return nil
	}
	fn := m.rankFn
	if fn == nil {
		fn = m.fn
	}
	return batchLoop(fn, query, flat, out)
}

// batchSize validates the flat-layout arguments shared by DistanceBatch and
// RankBatch and returns the number of candidate vectors.
func batchSize(query, flat []float32, out []float64) (int, error) {
	if len(query) == 0 {
		return 0, errors.New("query must not be empty")
	}
	if len(flat)%len(query) != 0 {
		return 0, fmt.Errorf("flat length %d is not a multiple of the query length %d",
			len(flat), len(query))
	}
	n := len(flat) / len(query)
	if len(out) != n {
		return 0, fmt.Errorf("out length %d does not match the candidate count %d", len(out), n)
	}
	return n, nil
}

// batchLoop computes a batch one candidate at a time through the given
// per-pair function, for metrics without a batch kernel.
func batchLoop(fn DistanceFunc, query, flat []float32, out []float64) error {
	dim := len(query)
	for i := range out {
		d, err := fn(query, flat[i*dim:(i+1)*dim])
		if err != nil {
			return err
		}
		out[i] = d
	}
	return nil
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
	// Euclidean is the Euclidean (L2) distance. Its rank distance is the
	// squared Euclidean distance, so index-internal comparisons skip the
	// square root.
	Euclidean = Metric{
		name: "euclidean", fn: euclidean, rankFn: squaredEuclidean, fromRank: math.Sqrt,
		batchFn: euclideanBatch, rankBatch: squaredEuclideanBatch,
	}
	// SquaredEuclidean is the squared Euclidean distance.
	SquaredEuclidean = Metric{name: "squared_euclidean", fn: squaredEuclidean, batchFn: squaredEuclideanBatch}
	// Manhattan is the Manhattan (L1) distance.
	Manhattan = Metric{name: "manhattan", fn: manhattan, batchFn: manhattanBatch}
	// Cosine is the cosine distance. Vectors are normalized when stored.
	Cosine = Metric{name: "cosine", fn: cosineDistance, normalize: true, batchFn: cosineDistanceBatch}
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

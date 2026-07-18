package vectors

import "math"

// Euclidean64 computes the L2 distance between two float64 vectors.
// Returns math.Inf(1) if vectors are empty or have different lengths.
func Euclidean64(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return math.Inf(1)
	}

	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

// ComputeCentroid64 returns the mean vector of the given points.
// Returns nil if points is empty.
func ComputeCentroid64(points [][]float64) []float64 {
	if len(points) == 0 {
		return nil
	}
	if len(points) == 1 {
		result := make([]float64, len(points[0]))
		copy(result, points[0])
		return result
	}

	dim := len(points[0])
	// Ragged input has no well-defined centroid; signal with nil.
	for _, p := range points {
		if len(p) != dim {
			return nil
		}
	}
	centroid := make([]float64, dim)

	for _, p := range points {
		for i := 0; i < dim; i++ {
			centroid[i] += p[i]
		}
	}

	n := float64(len(points))
	for i := range centroid {
		centroid[i] /= n
	}

	return centroid
}

// Dot64 returns the dot product of two float64 vectors.
// Returns 0 on length mismatch.
func Dot64(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

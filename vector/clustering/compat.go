package clustering

import "github.com/dotcommander/reliquary/vector"

// CosineDistance computes cosine distance from similarity.
func CosineDistance(a, b []float64) float64 {
	return vectors.CosineDistance64(a, b)
}

// EuclideanDistance computes L2 distance.
func EuclideanDistance(a, b []float64) float64 {
	return vectors.Euclidean64(a, b)
}

// ComputeCentroid computes the mean vector of points.
func ComputeCentroid(points [][]float64) []float64 {
	return vectors.ComputeCentroid64(points)
}

// DistanceFunc computes distance between two vectors.
// Kept for compatibility with existing HAC internals.
type DistanceFunc func(a, b []float64) float64

// DistanceMatrix computes pairwise distance over a metric.
func DistanceMatrix(points [][]float64, metric DistanceFunc) [][]float64 {
	n := len(points)
	if n == 0 {
		return nil
	}

	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		matrix[i][i] = 0
		for j := i + 1; j < n; j++ {
			d := metric(points[i], points[j])
			matrix[i][j] = d
			matrix[j][i] = d
		}
	}

	return matrix
}

// NormalizeVector normalizes a vector in place and returns it.
func NormalizeVector(vec []float64) []float64 {
	if len(vec) == 0 {
		return vec
	}

	vectors.Normalize64(vec)
	return vec
}

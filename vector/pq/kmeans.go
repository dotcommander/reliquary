package pq

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
)

// KMeansPlusPlus initializes k centroids using the k-means++ algorithm.
// This provides better initial centroids than random selection, leading to
// faster convergence and better clustering quality.
//
// Algorithm:
// 1. Choose first centroid uniformly at random
// 2. For each remaining centroid:
//   - Compute D(x)^2 = distance to nearest existing centroid
//   - Choose next centroid with probability proportional to D(x)^2
func KMeansPlusPlus(vectors [][]float32, k int) ([][]float32, error) {
	if _, err := validateKMeansVectors(vectors); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if k > len(vectors) {
		return nil, errors.New("k cannot exceed number of vectors")
	}

	centroids := make([][]float32, k)

	// Choose first centroid uniformly at random
	firstIdx := rand.IntN(len(vectors))
	centroids[0] = copyVector(vectors[firstIdx])

	// Track squared distances to nearest centroid
	distances := make([]float32, len(vectors))
	for i := range distances {
		distances[i] = math.MaxFloat32
	}

	// Choose remaining centroids
	for c := 1; c < k; c++ {
		// Update distances to include the new centroid
		lastCentroid := centroids[c-1]
		var totalDist float64
		for i, v := range vectors {
			dist := squaredL2Distance(v, lastCentroid)
			if dist < distances[i] {
				distances[i] = dist
			}
			totalDist += float64(distances[i])
		}

		// Choose next centroid with probability proportional to D(x)^2
		threshold := rand.Float64() * totalDist
		var cumulative float64
		chosen := len(vectors) - 1 // Default to last if rounding issues
		for i, dist := range distances {
			cumulative += float64(dist)
			if cumulative >= threshold {
				chosen = i
				break
			}
		}
		centroids[c] = copyVector(vectors[chosen])
	}

	return centroids, nil
}

// KMeans clusters vectors into k centroids using Lloyd's algorithm.
// Uses k-means++ initialization for better results.
//
// Parameters:
//   - vectors: the data points to cluster
//   - k: number of clusters
//   - maxIter: maximum iterations (typically 20-50)
//
// Returns the k centroids. Empty clusters are handled by reinitializing
// from the farthest point.
func KMeans(vectors [][]float32, k int, maxIter int) ([][]float32, error) {
	dim, err := validateKMeansVectors(vectors)
	if err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if k > len(vectors) {
		// If we have fewer vectors than k, just return copies of all vectors
		// padded with duplicates
		centroids := make([][]float32, k)
		for i := 0; i < k; i++ {
			centroids[i] = copyVector(vectors[i%len(vectors)])
		}
		return centroids, nil
	}
	if maxIter <= 0 {
		return nil, errors.New("maxIter must be positive")
	}

	// Initialize with k-means++
	centroids, err := KMeansPlusPlus(vectors, k)
	if err != nil {
		return nil, err
	}

	// Track assignments
	assignments := make([]int, len(vectors))
	clusterSums := make([][]float64, k)
	clusterCounts := make([]int, k)
	for i := range clusterSums {
		clusterSums[i] = make([]float64, dim)
	}

	for iter := 0; iter < maxIter; iter++ {
		// Reset accumulators
		for i := range clusterSums {
			for d := range clusterSums[i] {
				clusterSums[i][d] = 0
			}
			clusterCounts[i] = 0
		}

		// Assign vectors to nearest centroid
		changed := false
		for i, v := range vectors {
			nearest := findNearest(v, centroids)
			if nearest != assignments[i] {
				changed = true
				assignments[i] = nearest
			}

			// Accumulate for centroid update
			for d := range v {
				clusterSums[nearest][d] += float64(v[d])
			}
			clusterCounts[nearest]++
		}

		// Early termination if no changes
		if !changed && iter > 0 {
			break
		}

		// Update centroids
		for c := 0; c < k; c++ {
			if clusterCounts[c] == 0 {
				// Handle empty cluster: reinitialize with farthest point
				centroids[c] = findFarthestPoint(vectors, centroids)
			} else {
				for d := 0; d < dim; d++ {
					centroids[c][d] = float32(clusterSums[c][d] / float64(clusterCounts[c]))
				}
			}
		}
	}

	return centroids, nil
}

// findNearest returns the index of the nearest centroid to the vector.
func findNearest(v []float32, centroids [][]float32) int {
	minDist := float32(math.MaxFloat32)
	minIdx := 0
	for i, c := range centroids {
		dist := squaredL2Distance(v, c)
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	return minIdx
}

// findFarthestPoint returns the point farthest from any centroid.
// Used to reinitialize empty clusters.
func findFarthestPoint(vectors [][]float32, centroids [][]float32) []float32 {
	maxDist := float32(0)
	farthestIdx := 0

	for i, v := range vectors {
		// Find distance to nearest centroid
		minDist := float32(math.MaxFloat32)
		for _, c := range centroids {
			dist := squaredL2Distance(v, c)
			if dist < minDist {
				minDist = dist
			}
		}
		if minDist > maxDist {
			maxDist = minDist
			farthestIdx = i
		}
	}

	return copyVector(vectors[farthestIdx])
}

// copyVector creates a deep copy of a vector.
func copyVector(v []float32) []float32 {
	result := make([]float32, len(v))
	copy(result, v)
	return result
}

// MiniBatchKMeans performs k-means with mini-batches for large datasets.
// More memory-efficient than full k-means while providing similar quality.
//
// Parameters:
//   - vectors: the data points to cluster
//   - k: number of clusters
//   - batchSize: number of samples per iteration
//   - maxIter: maximum iterations
func MiniBatchKMeans(vectors [][]float32, k int, batchSize int, maxIter int) ([][]float32, error) {
	dim, err := validateKMeansVectors(vectors)
	if err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if batchSize <= 0 {
		return nil, errors.New("batchSize must be positive")
	}
	if batchSize > len(vectors) {
		batchSize = len(vectors)
	}

	// Initialize with k-means++
	centroids, err := KMeansPlusPlus(vectors, k)
	if err != nil {
		return nil, err
	}

	// Track per-centroid update counts for averaging
	centroidCounts := make([]int, k)

	for iter := 0; iter < maxIter; iter++ {
		// Sample a mini-batch
		batch := sampleBatch(vectors, batchSize)

		// Find nearest centroids for batch
		for _, v := range batch {
			nearest := findNearest(v, centroids)

			// Online update: move centroid towards the point
			// Using learning rate = 1 / (count + 1)
			centroidCounts[nearest]++
			lr := 1.0 / float32(centroidCounts[nearest])
			for d := 0; d < dim; d++ {
				centroids[nearest][d] += lr * (v[d] - centroids[nearest][d])
			}
		}
	}

	return centroids, nil
}

func validateKMeansVectors(vectors [][]float32) (int, error) {
	if len(vectors) == 0 {
		return 0, errors.New("no vectors provided")
	}
	dim := len(vectors[0])
	if dim == 0 {
		return 0, errors.New("vectors must have positive dimension")
	}
	for i, vector := range vectors {
		if len(vector) != dim {
			return 0, fmt.Errorf("vector %d has dimension %d, expected %d", i, len(vector), dim)
		}
		for j, value := range vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return 0, fmt.Errorf("vector %d has non-finite value at index %d: %v", i, j, value)
			}
		}
	}
	return dim, nil
}

// sampleBatch randomly samples vectors without replacement.
func sampleBatch(vectors [][]float32, size int) [][]float32 {
	if size >= len(vectors) {
		return vectors
	}

	// Fisher-Yates partial shuffle
	indices := make([]int, len(vectors))
	for i := range indices {
		indices[i] = i
	}

	batch := make([][]float32, size)
	for i := 0; i < size; i++ {
		j := i + rand.IntN(len(vectors)-i)
		indices[i], indices[j] = indices[j], indices[i]
		batch[i] = vectors[indices[i]]
	}

	return batch
}

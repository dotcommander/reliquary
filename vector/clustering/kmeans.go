package clustering

import "github.com/dotcommander/reliquary/vector"

// KMeansConfig holds configuration for K-means clustering.
type KMeansConfig struct {
	K             int     // number of clusters
	MaxIterations int     // maximum iterations (default: 100)
	Tolerance     float64 // convergence tolerance (default: 1e-4)
	Seed          int64   // random seed for initialization (0 = deterministic default)
}

// DefaultKMeansConfig returns default K-means configuration.
func DefaultKMeansConfig() KMeansConfig {
	return KMeansConfig{
		K:             0, // will be auto-selected via silhouette if 0
		MaxIterations: 100,
		Tolerance:     1e-4,
		Seed:          0,
	}
}

// KMeansResult holds the result of K-means clustering.
type KMeansResult struct {
	Assignments []int       // cluster ID for each point
	Centroids   [][]float64 // cluster centroids
	K           int         // number of clusters
	Iterations  int         // iterations until convergence
	Converged   bool        // whether algorithm converged
}

// KMeans performs K-means clustering with K-means++ initialization.
func KMeans(embeddings [][]float64, cfg KMeansConfig) *KMeansResult {
	result := vectors.KMeans64(embeddings, vectors.KMeans64Config{
		K:             cfg.K,
		MaxIterations: cfg.MaxIterations,
		Tolerance:     cfg.Tolerance,
		Seed:          cfg.Seed,
	})

	return &KMeansResult{
		Assignments: result.Assignments,
		Centroids:   result.Centroids,
		K:           result.K,
		Iterations:  result.Iterations,
		Converged:   result.Converged,
	}
}

func computeClusterCentroids(embeddings [][]float64, assignments []int, k int) [][]float64 {
	return vectors.ComputeClusterCentroids64(embeddings, assignments, k)
}

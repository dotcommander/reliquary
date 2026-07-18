package clustering

import (
	"testing"
)

func TestKMeans(t *testing.T) {
	t.Parallel()
	// Create clearly separable clusters
	embeddings := [][]float64{
		// Cluster 1: around [1, 0]
		{0.9, 0.1},
		{1.0, 0.0},
		{0.95, 0.05},
		// Cluster 2: around [0, 1]
		{0.1, 0.9},
		{0.0, 1.0},
		{0.05, 0.95},
	}

	result := KMeans(embeddings, KMeansConfig{
		K:             2,
		MaxIterations: 100,
		Tolerance:     1e-4,
		Seed:          42,
	})

	if result.K != 2 {
		t.Errorf("expected K=2, got K=%d", result.K)
	}

	if len(result.Assignments) != 6 {
		t.Errorf("expected 6 assignments, got %d", len(result.Assignments))
	}

	// Check that first 3 points are in the same cluster
	cluster0 := result.Assignments[0]
	for i := 1; i < 3; i++ {
		if result.Assignments[i] != cluster0 {
			t.Errorf("point %d expected in cluster %d, got %d", i, cluster0, result.Assignments[i])
		}
	}

	// Check that last 3 points are in the same cluster (different from first)
	cluster1 := result.Assignments[3]
	if cluster1 == cluster0 {
		t.Errorf("expected two different clusters, got same cluster %d", cluster0)
	}
	for i := 4; i < 6; i++ {
		if result.Assignments[i] != cluster1 {
			t.Errorf("point %d expected in cluster %d, got %d", i, cluster1, result.Assignments[i])
		}
	}
}

func TestComputeClusterCentroidsRagged(t *testing.T) {
	t.Parallel()
	// dim is taken from embeddings[0]; a longer later embedding must not panic.
	dim := 2
	k := 2
	embeddings := [][]float64{
		{0.9, 0.1},
		{1.0, 0.0, 99.0}, // longer than dim — must be skipped, not panic
		{0.1, 0.9},
	}
	assignments := []int{0, 1, 1}

	centroids := computeClusterCentroids(embeddings, assignments, k)

	if len(centroids) != k {
		t.Fatalf("expected %d centroids, got %d", k, len(centroids))
	}
	for i, c := range centroids {
		if len(c) != dim {
			t.Errorf("centroid %d: expected length %d, got %d", i, dim, len(c))
		}
	}
}

func TestKMeansEmpty(t *testing.T) {
	t.Parallel()
	result := KMeans(nil, DefaultKMeansConfig())
	if result.K != 0 {
		t.Errorf("expected K=0 for empty input, got K=%d", result.K)
	}
}

func TestKMeansSinglePoint(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{{1, 0, 0}}
	result := KMeans(embeddings, KMeansConfig{K: 1})

	if result.K != 1 {
		t.Errorf("expected K=1, got K=%d", result.K)
	}
	if len(result.Assignments) != 1 {
		t.Errorf("expected 1 assignment, got %d", len(result.Assignments))
	}
}

func TestKMeansConvergence(t *testing.T) {
	t.Parallel()
	// Simple case that should converge quickly
	embeddings := [][]float64{
		{1, 0},
		{0, 1},
	}

	result := KMeans(embeddings, KMeansConfig{
		K:             2,
		MaxIterations: 100,
	})

	if !result.Converged {
		t.Errorf("expected convergence, but did not converge in %d iterations", result.Iterations)
	}
}

func TestKMeansKGreaterThanN(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{
		{1, 0},
		{0, 1},
	}

	result := KMeans(embeddings, KMeansConfig{K: 5}) // K > N

	// Should cap K at N
	if result.K > 2 {
		t.Errorf("expected K <= 2 (number of points), got K=%d", result.K)
	}
}

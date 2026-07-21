package clustering

import (
	"math"
	"strings"
	"testing"
)

func TestServiceKMeans(t *testing.T) {
	t.Parallel()
	svc := NewClusterService("kmeans")

	embeddings := [][]float64{
		{1, 0}, {0.9, 0.1},
		{0, 1}, {0.1, 0.9},
	}

	result, err := svc.Cluster(embeddings, ClusterOptions{
		K: 2,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.K != 2 {
		t.Errorf("expected K=2, got K=%d", result.K)
	}

	if len(result.Assignments) != 4 {
		t.Errorf("expected 4 assignments, got %d", len(result.Assignments))
	}
}

func TestServiceHAC(t *testing.T) {
	t.Parallel()
	svc := NewClusterService("hac")

	embeddings := [][]float64{
		{1, 0}, {0.9, 0.1},
		{0, 1}, {0.1, 0.9},
	}

	result, err := svc.Cluster(embeddings, ClusterOptions{
		K:       2,
		Linkage: "average",
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.K != 2 {
		t.Errorf("expected K=2, got K=%d", result.K)
	}
}

func TestServiceGreedy(t *testing.T) {
	t.Parallel()
	svc := NewClusterService("greedy")

	embeddings := [][]float64{
		{1, 0}, {0.95, 0.05}, {0.9, 0.1}, // similar
		{0, 1}, {0.05, 0.95}, {0.1, 0.9}, // similar
	}

	result, err := svc.Cluster(embeddings, ClusterOptions{
		TargetMax: 3,
		HardMax:   5,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should create clusters (exact number depends on similarity)
	if result.K < 1 {
		t.Errorf("expected at least 1 cluster, got K=%d", result.K)
	}

	if len(result.Assignments) != 6 {
		t.Errorf("expected 6 assignments, got %d", len(result.Assignments))
	}
}

func TestServiceAutoK(t *testing.T) {
	t.Parallel()
	svc := NewClusterService("kmeans")

	embeddings := [][]float64{
		{1, 0}, {0.95, 0.05},
		{0, 1}, {0.05, 0.95},
		{0.5, 0.5}, {0.55, 0.45},
	}

	result, err := svc.Cluster(embeddings, ClusterOptions{
		K:    0, // auto-select
		MaxK: 5,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.K < 2 || result.K > 5 {
		t.Errorf("expected 2 <= K <= 5, got K=%d", result.K)
	}

	// Should have silhouette score
	if result.Silhouette < -1 || result.Silhouette > 1 {
		t.Errorf("silhouette out of range: %v", result.Silhouette)
	}
}

func TestServiceEmpty(t *testing.T) {
	t.Parallel()
	svc := NewClusterService("kmeans")

	result, err := svc.Cluster(nil, DefaultClusterOptions())

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.K != 0 {
		t.Errorf("expected K=0 for empty input, got K=%d", result.K)
	}
}

func TestServiceRejectsMalformedEmbeddingsForEveryAlgorithm(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		embeddings [][]float64
		wantError  string
	}{
		{name: "zero-dimensional", embeddings: [][]float64{{}}, wantError: "embedding 0 has zero dimensions"},
		{name: "ragged", embeddings: [][]float64{{1, 0}, {1}}, wantError: "embedding 1 has dimension 1, want 2"},
		{name: "nan", embeddings: [][]float64{{1, math.NaN()}}, wantError: "embedding 0 value 1 is non-finite"},
		{name: "positive infinity", embeddings: [][]float64{{math.Inf(1), 0}}, wantError: "embedding 0 value 0 is non-finite"},
		{name: "negative infinity", embeddings: [][]float64{{math.Inf(-1), 0}}, wantError: "embedding 0 value 0 is non-finite"},
	}
	for _, algorithm := range []string{"greedy", "kmeans", "hac"} {
		algorithm := algorithm
		for _, tc := range cases {
			tc := tc
			t.Run(algorithm+"/"+tc.name, func(t *testing.T) {
				t.Parallel()
				result, err := NewClusterService(algorithm).Cluster(tc.embeddings, ClusterOptions{K: 2})
				if err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("Cluster() error = %v, want containing %q", err, tc.wantError)
				}
				if result != nil {
					t.Fatalf("Cluster() result = %#v, want nil", result)
				}
			})
		}
	}
}

func TestServiceGreedyWithSizeConstraints(t *testing.T) {
	t.Parallel()
	svc := NewClusterService("greedy")

	// 4 similar embeddings
	embeddings := [][]float64{
		{1, 0}, {0.99, 0.01}, {0.98, 0.02}, {0.97, 0.03},
	}

	// File sizes that will trigger size constraint
	fileSizes := []int64{
		30 * 1024, // 30KB
		30 * 1024, // 30KB
		30 * 1024, // 30KB
		30 * 1024, // 30KB
	}

	result, err := svc.Cluster(embeddings, ClusterOptions{
		Algorithm: "greedy",
		TargetMax: 10,
		HardMax:   10,
		MaxSizeKB: 80, // 80KB limit
		FileSizes: fileSizes,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should create at least 2 clusters due to size constraint
	if result.K < 2 {
		t.Errorf("expected at least 2 clusters due to size constraint, got K=%d", result.K)
	}
}

func TestServiceAlgorithmOverride(t *testing.T) {
	t.Parallel()
	// Create service with default algorithm
	svc := NewClusterService("greedy")

	embeddings := [][]float64{
		{1, 0}, {0.9, 0.1},
		{0, 1}, {0.1, 0.9},
	}

	// Override algorithm in options
	result, err := svc.Cluster(embeddings, ClusterOptions{
		Algorithm: "kmeans", // override
		K:         2,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.K != 2 {
		t.Errorf("expected K=2, got K=%d", result.K)
	}
}

func TestNewClusterServiceDefault(t *testing.T) {
	t.Parallel()
	svc := NewClusterService("")

	// Should use greedy by default
	embeddings := [][]float64{
		{1, 0}, {0, 1},
	}

	result, err := svc.Cluster(embeddings, ClusterOptions{})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.K < 1 {
		t.Errorf("expected at least 1 cluster")
	}
}

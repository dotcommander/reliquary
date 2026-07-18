package clustering

import (
	"testing"
)

func TestHAC(t *testing.T) {
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

	result := HAC(embeddings, HACConfig{
		K:       2,
		Linkage: LinkageAverage,
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

func TestHACLinkageMethods(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{
		{1, 0},
		{0.9, 0.1},
		{0, 1},
		{0.1, 0.9},
	}

	linkages := []Linkage{LinkageSingle, LinkageComplete, LinkageAverage}

	for _, linkage := range linkages {
		t.Run(string(linkage), func(t *testing.T) {
			result := HAC(embeddings, HACConfig{
				K:       2,
				Linkage: linkage,
			})

			if result.K != 2 {
				t.Errorf("linkage %s: expected K=2, got K=%d", linkage, result.K)
			}

			if len(result.Assignments) != 4 {
				t.Errorf("linkage %s: expected 4 assignments, got %d", linkage, len(result.Assignments))
			}
		})
	}
}

func TestHACEmpty(t *testing.T) {
	t.Parallel()
	result := HAC(nil, DefaultHACConfig())
	if result.K != 0 {
		t.Errorf("expected K=0 for empty input, got K=%d", result.K)
	}
}

func TestHACSinglePoint(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{{1, 0, 0}}
	result := HAC(embeddings, HACConfig{K: 1})

	if result.K != 1 {
		t.Errorf("expected K=1, got K=%d", result.K)
	}
	if len(result.Assignments) != 1 {
		t.Errorf("expected 1 assignment, got %d", len(result.Assignments))
	}
}

func TestHACDendrogram(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{
		{1, 0},
		{0.9, 0.1},
		{0, 1},
	}

	result := HAC(embeddings, HACConfig{
		K:       1, // merge all
		Linkage: LinkageAverage,
	})

	// Should have N-1 merge steps
	expectedMerges := len(embeddings) - 1
	if len(result.Dendrogram) != expectedMerges {
		t.Errorf("expected %d merge steps, got %d", expectedMerges, len(result.Dendrogram))
	}

	// Merge distances should be non-decreasing (for agglomerative)
	for i := 1; i < len(result.Dendrogram); i++ {
		if result.Dendrogram[i].Distance < result.Dendrogram[i-1].Distance {
			t.Errorf("dendrogram distances not monotonic: step %d (%v) < step %d (%v)",
				i, result.Dendrogram[i].Distance, i-1, result.Dendrogram[i-1].Distance)
		}
	}
}

func TestHACComputeMergedCentroid(t *testing.T) {
	t.Parallel()
	// (a) uniform 3-point input returns the arithmetic mean
	embeddings := [][]float64{
		{1, 2, 3},
		{3, 4, 5},
		{5, 6, 7},
	}
	got := computeMergedCentroid(embeddings, []int{0, 1, 2})
	want := []float64{3, 4, 5}
	if len(got) != len(want) {
		t.Fatalf("expected centroid of length %d, got %d", len(want), len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("centroid[%d]: expected %v, got %v", i, want[i], got[i])
		}
	}

	// (b) ragged members returns nil without panic
	ragged := [][]float64{
		{1, 2, 3},
		{4, 5},
	}
	if r := computeMergedCentroid(ragged, []int{0, 1}); r != nil {
		t.Errorf("expected nil for ragged input, got %v", r)
	}
}

func TestHACKGreaterThanN(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{
		{1, 0},
		{0, 1},
	}

	result := HAC(embeddings, HACConfig{K: 5}) // K > N

	// Should cap K at N
	if result.K > 2 {
		t.Errorf("expected K <= 2 (number of points), got K=%d", result.K)
	}
}

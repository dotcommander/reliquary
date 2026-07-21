package clustering

import (
	"math"
	"testing"
)

func TestSilhouetteCoefficient(t *testing.T) {
	t.Parallel()
	// Two well-separated clusters
	embeddings := [][]float64{
		{1, 0}, {0.9, 0.1}, // cluster 0
		{0, 1}, {0.1, 0.9}, // cluster 1
	}
	assignments := []int{0, 0, 1, 1}

	// Points in well-separated clusters should have high silhouette
	for i := range embeddings {
		coef := SilhouetteCoefficient(i, embeddings, assignments)
		if coef < 0.5 {
			t.Errorf("point %d: expected silhouette > 0.5 for well-separated clusters, got %v", i, coef)
		}
	}
}

func TestSilhouetteSinglePointCluster(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{
		{1, 0},
		{0, 1},
	}
	assignments := []int{0, 1} // each point in its own cluster

	// Single-point clusters should have silhouette = 0
	for i := range embeddings {
		coef := SilhouetteCoefficient(i, embeddings, assignments)
		if coef != 0 {
			t.Errorf("point %d: expected silhouette = 0 for single-point cluster, got %v", i, coef)
		}
	}
}

func TestAverageSilhouetteScore(t *testing.T) {
	t.Parallel()
	// Well-separated clusters
	embeddings := [][]float64{
		{1, 0}, {0.95, 0.05}, {0.9, 0.1},
		{0, 1}, {0.05, 0.95}, {0.1, 0.9},
	}
	assignments := []int{0, 0, 0, 1, 1, 1}

	score := AverageSilhouetteScore(embeddings, assignments)

	// Well-separated clusters should have positive average silhouette
	if score <= 0 {
		t.Errorf("expected positive silhouette for well-separated clusters, got %v", score)
	}
	if score > 1 {
		t.Errorf("silhouette should be <= 1, got %v", score)
	}
}

func TestFindOptimalK(t *testing.T) {
	t.Parallel()
	// Create 3 clear clusters
	embeddings := [][]float64{
		// Cluster around [1, 0, 0]
		{1, 0, 0}, {0.95, 0.05, 0}, {0.9, 0.1, 0},
		// Cluster around [0, 1, 0]
		{0, 1, 0}, {0.05, 0.95, 0}, {0.1, 0.9, 0},
		// Cluster around [0, 0, 1]
		{0, 0, 1}, {0, 0.05, 0.95}, {0, 0.1, 0.9},
	}

	result := FindOptimalK(embeddings, SilhouetteConfig{
		MinK:      2,
		MaxK:      5,
		Algorithm: "kmeans",
	})

	// Should find k=3 as optimal (or at least a reasonable k)
	if result.BestK < 2 || result.BestK > 5 {
		t.Errorf("expected 2 <= BestK <= 5, got %d", result.BestK)
	}

	// Should have scores for each k tried
	expectedScores := 4 // k = 2, 3, 4, 5
	if len(result.Scores) != expectedScores {
		t.Errorf("expected %d scores, got %d", expectedScores, len(result.Scores))
	}

	// Best score should be in valid range
	if result.BestScore < -1 || result.BestScore > 1 {
		t.Errorf("silhouette score out of range [-1, 1]: %v", result.BestScore)
	}

	// Should return assignments
	if len(result.Assignments) != len(embeddings) {
		t.Errorf("expected %d assignments, got %d", len(embeddings), len(result.Assignments))
	}
}

func TestFindOptimalKWithHAC(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{
		{1, 0}, {0.9, 0.1},
		{0, 1}, {0.1, 0.9},
	}

	result := FindOptimalK(embeddings, SilhouetteConfig{
		MinK:      2,
		MaxK:      3,
		Algorithm: "hac",
	})

	if result.BestK < 2 {
		t.Errorf("expected BestK >= 2, got %d", result.BestK)
	}
	if len(result.Scores) != 2 || len(result.KValues) != 2 || len(result.Assignments) != len(embeddings) || len(result.Centroids) != result.BestK {
		t.Fatalf("incoherent feasible HAC metadata: %+v", result)
	}
	if result.KValues[0] != 2 || result.KValues[1] != 3 {
		t.Fatalf("KValues = %v, want [2 3]", result.KValues)
	}
}

func TestFindOptimalKWithHACImpossibleRange(t *testing.T) {
	t.Parallel()

	result := FindOptimalK([][]float64{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}, SilhouetteConfig{
		MinK:      4,
		MaxK:      10,
		Algorithm: "hac",
	})
	if result.BestK != 0 || result.BestScore != 0 || result.Scores != nil || result.KValues != nil || result.Assignments != nil || result.Centroids != nil {
		t.Fatalf("FindOptimalK impossible range = %+v, want empty result", result)
	}
}

func TestFindOptimalKSmallDataset(t *testing.T) {
	t.Parallel()
	// Too few points to cluster meaningfully
	embeddings := [][]float64{
		{1, 0},
	}

	result := FindOptimalK(embeddings, DefaultSilhouetteConfig())

	if result.BestK != 1 {
		t.Errorf("expected BestK = 1 for single point, got %d", result.BestK)
	}
}

func TestClusterSilhouetteScores(t *testing.T) {
	t.Parallel()
	embeddings := [][]float64{
		{1, 0}, {0.9, 0.1}, // cluster 0 - similar
		{0, 1}, {0.1, 0.9}, // cluster 1 - similar
	}
	assignments := []int{0, 0, 1, 1}

	scores := ClusterSilhouetteScores(embeddings, assignments)

	if len(scores) != 2 {
		t.Errorf("expected 2 cluster scores, got %d", len(scores))
	}

	// Both clusters should have positive scores
	for cluster, score := range scores {
		if score <= 0 {
			t.Errorf("cluster %d: expected positive score, got %v", cluster, score)
		}
		if math.IsNaN(score) {
			t.Errorf("cluster %d: score is NaN", cluster)
		}
	}
}

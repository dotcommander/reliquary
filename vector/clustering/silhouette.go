package clustering

import "github.com/dotcommander/reliquary/vector"

// SilhouetteConfig holds configuration for silhouette-based auto-k selection.
type SilhouetteConfig struct {
	MinK      int    // minimum k to try (default: 2)
	MaxK      int    // maximum k to try (default: 20)
	Algorithm string // "kmeans" or "hac" (default: "kmeans")
}

// DefaultSilhouetteConfig returns default silhouette configuration.
func DefaultSilhouetteConfig() SilhouetteConfig {
	return SilhouetteConfig{
		MinK:      2,
		MaxK:      20,
		Algorithm: "kmeans",
	}
}

// SilhouetteResult holds the result of silhouette analysis.
type SilhouetteResult struct {
	BestK       int       // best k (tie-break: smaller k)
	BestScore   float64   // silhouette score at best k
	Scores      []float64 // silhouette scores for each k tried
	KValues     []int     // k values tried
	Assignments []int     // cluster assignments at best k
	Centroids   [][]float64
}

// SilhouetteCoefficient computes the silhouette coefficient for a single sample.
func SilhouetteCoefficient(pointIdx int, embeddings [][]float64, assignments []int) float64 {
	return vectors.SilhouetteCoefficient64(pointIdx, embeddings, assignments)
}

// AverageSilhouetteScore computes the average silhouette score for a clustering.
func AverageSilhouetteScore(embeddings [][]float64, assignments []int) float64 {
	return vectors.AverageSilhouetteScore64(embeddings, assignments)
}

// FindOptimalK sweeps k from MinK to min(MaxK, N-1) and returns the best k.
// Tie-break: smaller k wins (simpler model).
func FindOptimalK(embeddings [][]float64, cfg SilhouetteConfig) *SilhouetteResult {
	algorithm := cfg.Algorithm
	if algorithm == "" {
		algorithm = "kmeans"
	}
	if algorithm != "hac" {
		bestK, bestScore, assignments, centroids, scores, kValues := vectors.FindOptimalK64(embeddings, cfg.MinK, cfg.MaxK)
		return &SilhouetteResult{
			BestK:       bestK,
			BestScore:   bestScore,
			Scores:      scores,
			KValues:     kValues,
			Assignments: assignments,
			Centroids:   centroids,
		}
	}

	n := len(embeddings)
	if n <= 2 {
		return &SilhouetteResult{
			BestK:       1,
			BestScore:   0,
			Scores:      []float64{0},
			KValues:     []int{1},
			Assignments: make([]int, n),
		}
	}

	minK := cfg.MinK
	if minK < 2 {
		minK = 2
	}

	maxK := cfg.MaxK
	if maxK <= 0 {
		maxK = 20
	}
	if maxK > n-1 {
		maxK = n - 1
	}
	if maxK < minK {
		maxK = minK
	}

	bestK := 0
	bestScore := -2.0
	var bestAssignments []int
	var bestCentroids [][]float64
	var scores []float64
	var kValues []int

	for k := minK; k <= maxK; k++ {
		result := HAC(embeddings, HACConfig{K: k, Linkage: LinkageAverage})
		score := AverageSilhouetteScore(embeddings, result.Assignments)
		scores = append(scores, score)
		kValues = append(kValues, k)
		if score > bestScore {
			bestScore = score
			bestK = k
			bestAssignments = result.Assignments
			bestCentroids = result.Centroids
		}
	}

	return &SilhouetteResult{
		BestK:       bestK,
		BestScore:   bestScore,
		Scores:      scores,
		KValues:     kValues,
		Assignments: bestAssignments,
		Centroids:   bestCentroids,
	}
}

// ClusterSilhouetteScores computes silhouette score per cluster.
func ClusterSilhouetteScores(embeddings [][]float64, assignments []int) map[int]float64 {
	return vectors.ClusterSilhouetteScores64(embeddings, assignments)
}

package vectors

import (
	"math"
	"sort"
	"testing"
)

func TestKMeans64_DeterministicWithSeedZero(t *testing.T) {
	t.Parallel()

	points := [][]float64{
		{0, 0},
		{0.1, 0.1},
		{-0.1, 0.1},
		{10, 10},
		{10.1, 10.1},
		{9.9, 10.1},
		{20, 20},
		{20.1, 20.1},
		{19.9, 20.1},
	}

	cfg := KMeans64Config{K: 3, MaxIterations: 200, Tolerance: 1e-8, Seed: 0}

	resultA := KMeans64(points, cfg)
	resultB := KMeans64(points, cfg)

	if !resultA.Converged {
		t.Fatal("resultA.Converged = false, want true")
	}
	if !resultB.Converged {
		t.Fatal("resultB.Converged = false, want true")
	}
	if resultA.K != resultB.K {
		t.Fatalf("K mismatch: %d vs %d", resultA.K, resultB.K)
	}
	if resultA.Iterations != resultB.Iterations {
		t.Fatalf("Iterations mismatch: %d vs %d", resultA.Iterations, resultB.Iterations)
	}
	if len(resultA.Assignments) != len(points) || len(resultB.Assignments) != len(points) {
		t.Fatalf("assignment length mismatch: %d, %d, want %d", len(resultA.Assignments), len(resultB.Assignments), len(points))
	}
	if len(resultA.Centroids) != 3 || len(resultB.Centroids) != 3 {
		t.Fatalf("centroid count mismatch: %d, %d, want %d", len(resultA.Centroids), len(resultB.Centroids), 3)
	}

	sortedA := sortedCentroidsByFirstCoord(resultA.Centroids)
	sortedB := sortedCentroidsByFirstCoord(resultB.Centroids)
	for i := range sortedA {
		if len(sortedA[i]) != len(sortedB[i]) {
			t.Fatalf("centroid dim mismatch at index %d: %d vs %d", i, len(sortedA[i]), len(sortedB[i]))
		}
		for j := range sortedA[i] {
			if math.Abs(sortedA[i][j]-sortedB[i][j]) > 1e-6 {
				t.Fatalf("centroid mismatch at [%d][%d]: %g vs %g", i, j, sortedA[i][j], sortedB[i][j])
			}
		}
	}

	for i, assignment := range resultA.Assignments {
		if assignment != resultB.Assignments[i] {
			t.Fatalf("assignment mismatch at index %d: %d vs %d", i, assignment, resultB.Assignments[i])
		}
	}
}

func TestComputeClusterCentroids64_SkipsOutOfRangeAssignments(t *testing.T) {
	t.Parallel()

	points := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
		{7, 8},
	}
	assignments := []int{0, 1, 5, -1}

	var centroids [][]float64
	func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				t.Fatalf("ComputeClusterCentroids64 panicked: %v", recovered)
			}
		}()
		centroids = ComputeClusterCentroids64(points, assignments, 2)
	}()

	if centroids == nil {
		t.Fatal("ComputeClusterCentroids64 returned nil")
	}
	if len(centroids) != 2 {
		t.Fatalf("centroid count mismatch: %d, want 2", len(centroids))
	}

	if len(centroids[0]) != 2 {
		t.Fatalf("centroid[0] dim mismatch: %d, want 2", len(centroids[0]))
	}
	if len(centroids[1]) != 2 {
		t.Fatalf("centroid[1] dim mismatch: %d, want 2", len(centroids[1]))
	}

	if math.Abs(centroids[0][0]-1) > 1e-6 || math.Abs(centroids[0][1]-2) > 1e-6 {
		t.Fatalf("centroid[0] = [%g, %g], want [1,2]", centroids[0][0], centroids[0][1])
	}
	if math.Abs(centroids[1][0]-3) > 1e-6 || math.Abs(centroids[1][1]-4) > 1e-6 {
		t.Fatalf("centroid[1] = [%g, %g], want [3,4]", centroids[1][0], centroids[1][1])
	}
}

func sortedCentroidsByFirstCoord(centroids [][]float64) [][]float64 {
	sorted := make([][]float64, len(centroids))
	for i, centroid := range centroids {
		copyCentroid := make([]float64, len(centroid))
		copy(copyCentroid, centroid)
		sorted[i] = copyCentroid
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i][0] < sorted[j][0]
	})
	return sorted
}

func TestSilhouetteScoresAndOptimalK64(t *testing.T) {
	t.Parallel()

	// Empty / single point edge cases
	if score := AverageSilhouetteScore64(nil, nil); score != 0 {
		t.Fatalf("AverageSilhouetteScore64(nil) = %v, want 0", score)
	}

	bestK, bestScore, assignments, centroids, scores, kValues := FindOptimalK64([][]float64{{1, 0}}, 2, 5)
	if bestK != 1 || bestScore != 0 || len(assignments) != 1 || centroids != nil || len(scores) != 1 || len(kValues) != 1 {
		t.Fatalf("FindOptimalK64 small input = %d, %v, %v, %v, %v, %v", bestK, bestScore, assignments, centroids, scores, kValues)
	}

	// Two distinct clusters
	points := [][]float64{
		{1, 0}, {0.9, 0.1}, {0.95, 0.05},
		{0, 1}, {0.1, 0.9}, {0.05, 0.95},
	}
	res := KMeans64(points, KMeans64Config{K: 2, Seed: 42})
	avgScore := AverageSilhouetteScore64(points, res.Assignments)
	if avgScore <= 0 {
		t.Fatalf("AverageSilhouetteScore64 = %v, want > 0", avgScore)
	}

	clusterScores := ClusterSilhouetteScores64(points, res.Assignments)
	if len(clusterScores) != 2 {
		t.Fatalf("ClusterSilhouetteScores64 count = %d, want 2", len(clusterScores))
	}

	bestK, bestScore, assignments, centroids, scores, kValues = FindOptimalK64(points, 2, 3)
	if bestK < 2 || bestScore <= 0 || len(assignments) != len(points) || len(centroids) == 0 || len(scores) == 0 || len(kValues) == 0 {
		t.Fatalf("FindOptimalK64 multi-point = bestK=%d score=%v assignments=%d centroids=%d", bestK, bestScore, len(assignments), len(centroids))
	}
}

package vectors

import (
	"math"
	"math/rand"
)

// KMeans64Config holds configuration for float64 K-means clustering.
type KMeans64Config struct {
	K             int
	MaxIterations int
	Tolerance     float64
	Seed          int64
}

// KMeans64Result holds the result of a float64 K-means clustering run.
type KMeans64Result struct {
	Assignments []int
	Centroids   [][]float64
	K           int
	Iterations  int
	Converged   bool
}

// KMeans64 performs K-means clustering with K-means++ initialization using
// cosine distance. It preserves the legacy float64 clustering semantics used
// by the higher-level clustering service.
func KMeans64(points [][]float64, cfg KMeans64Config) *KMeans64Result {
	n := len(points)
	if n == 0 {
		return &KMeans64Result{K: 0}
	}

	k := cfg.K
	if k <= 0 {
		k = 2
	}
	if k > n {
		k = n
	}

	maxIter := cfg.MaxIterations
	if maxIter <= 0 {
		maxIter = 100
	}

	tol := cfg.Tolerance
	if tol <= 0 {
		tol = 1e-4
	}

	var rng *rand.Rand
	if cfg.Seed != 0 {
		rng = rand.New(rand.NewSource(cfg.Seed))
	} else {
		rng = rand.New(rand.NewSource(int64(n*k) + 1))
	}

	centroids := kmeans64PlusPlusInit(points, k, rng)
	assignments := make([]int, n)

	var iterations int
	var converged bool
	for iter := 0; iter < maxIter; iter++ {
		iterations = iter + 1

		changed := false
		for i, point := range points {
			nearest := nearestCentroid64(point, centroids)
			if assignments[i] != nearest {
				assignments[i] = nearest
				changed = true
			}
		}

		if !changed {
			converged = true
			break
		}

		newCentroids := ComputeClusterCentroids64(points, assignments, k)

		maxMove := 0.0
		for i := range centroids {
			move := Euclidean64(centroids[i], newCentroids[i])
			if move > maxMove {
				maxMove = move
			}
		}

		centroids = newCentroids
		if maxMove < tol {
			converged = true
			break
		}
	}

	return &KMeans64Result{
		Assignments: assignments,
		Centroids:   centroids,
		K:           k,
		Iterations:  iterations,
		Converged:   converged,
	}
}

func kmeans64PlusPlusInit(points [][]float64, k int, rng *rand.Rand) [][]float64 {
	n := len(points)
	dim := len(points[0])
	centroids := make([][]float64, k)

	firstIdx := rng.Intn(n)
	centroids[0] = make([]float64, dim)
	copy(centroids[0], points[firstIdx])

	distances := make([]float64, n)
	for c := 1; c < k; c++ {
		totalDist := 0.0
		for i, point := range points {
			minDist := math.Inf(1)
			for j := 0; j < c; j++ {
				d := CosineDistance64(point, centroids[j])
				if d < minDist {
					minDist = d
				}
			}
			distances[i] = minDist * minDist
			totalDist += distances[i]
		}

		chosen := n - 1
		if totalDist > 0 {
			target := rng.Float64() * totalDist
			cumulative := 0.0
			for i := 0; i < n; i++ {
				cumulative += distances[i]
				if cumulative >= target {
					chosen = i
					break
				}
			}
		} else {
			chosen = rng.Intn(n)
		}

		centroids[c] = make([]float64, dim)
		copy(centroids[c], points[chosen])
	}

	return centroids
}

func nearestCentroid64(point []float64, centroids [][]float64) int {
	minDist := math.Inf(1)
	nearest := 0
	for i, centroid := range centroids {
		d := CosineDistance64(point, centroid)
		if d < minDist {
			minDist = d
			nearest = i
		}
	}
	return nearest
}

// ComputeClusterCentroids64 computes k centroids from assignments. Ragged
// points are skipped because a fixed-width centroid has no valid slot for them.
func ComputeClusterCentroids64(points [][]float64, assignments []int, k int) [][]float64 {
	if len(points) == 0 {
		return nil
	}

	dim := len(points[0])
	centroids := make([][]float64, k)
	counts := make([]int, k)
	for i := range centroids {
		centroids[i] = make([]float64, dim)
	}

	for i, point := range points {
		if len(point) != dim {
			continue
		}
		cluster := assignments[i]
		if cluster < 0 || cluster >= k {
			continue
		}
		counts[cluster]++
		for j, v := range point {
			centroids[cluster][j] += v
		}
	}

	for i := range centroids {
		if counts[i] == 0 {
			continue
		}
		for j := range centroids[i] {
			centroids[i][j] /= float64(counts[i])
		}
	}

	return centroids
}

// SilhouetteCoefficient64 computes the silhouette coefficient for one point.
func SilhouetteCoefficient64(pointIdx int, points [][]float64, assignments []int) float64 {
	if len(points) <= 1 {
		return 0
	}

	myCluster := assignments[pointIdx]

	var sumA float64
	countA := 0
	for j, point := range points {
		if j == pointIdx {
			continue
		}
		if assignments[j] == myCluster {
			sumA += CosineDistance64(points[pointIdx], point)
			countA++
		}
	}
	if countA == 0 {
		return 0
	}
	a := sumA / float64(countA)

	clusterDists := make(map[int]struct {
		sum   float64
		count int
	})
	for j, point := range points {
		if assignments[j] == myCluster {
			continue
		}
		entry := clusterDists[assignments[j]]
		entry.sum += CosineDistance64(points[pointIdx], point)
		entry.count++
		clusterDists[assignments[j]] = entry
	}

	b := math.Inf(1)
	for _, entry := range clusterDists {
		if entry.count == 0 {
			continue
		}
		avgDist := entry.sum / float64(entry.count)
		if avgDist < b {
			b = avgDist
		}
	}
	if math.IsInf(b, 1) {
		return 0
	}

	maxAB := math.Max(a, b)
	if maxAB == 0 {
		return 0
	}
	return (b - a) / maxAB
}

// AverageSilhouetteScore64 computes the average silhouette score.
func AverageSilhouetteScore64(points [][]float64, assignments []int) float64 {
	if len(points) == 0 {
		return 0
	}

	var sum float64
	for i := range points {
		sum += SilhouetteCoefficient64(i, points, assignments)
	}
	return sum / float64(len(points))
}

// FindOptimalK64 runs K-means for each candidate k and returns the best result.
func FindOptimalK64(points [][]float64, minK, maxK int) (bestK int, bestScore float64, assignments []int, centroids [][]float64, scores []float64, kValues []int) {
	n := len(points)
	if n <= 2 {
		return 1, 0, make([]int, n), nil, []float64{0}, []int{1}
	}
	if minK < 2 {
		minK = 2
	}
	if maxK <= 0 {
		maxK = 20
	}
	if maxK > n-1 {
		maxK = n - 1
	}
	if maxK < minK {
		maxK = minK
	}

	bestScore = -2
	for k := minK; k <= maxK; k++ {
		result := KMeans64(points, KMeans64Config{K: k, MaxIterations: 100, Tolerance: 1e-4})
		score := AverageSilhouetteScore64(points, result.Assignments)
		scores = append(scores, score)
		kValues = append(kValues, k)
		if score > bestScore {
			bestScore = score
			bestK = k
			assignments = result.Assignments
			centroids = result.Centroids
		}
	}

	return bestK, bestScore, assignments, centroids, scores, kValues
}

// ClusterSilhouetteScores64 computes the mean silhouette score per cluster.
func ClusterSilhouetteScores64(points [][]float64, assignments []int) map[int]float64 {
	scores := make(map[int]struct {
		sum   float64
		count int
	})

	for i := range points {
		cluster := assignments[i]
		coef := SilhouetteCoefficient64(i, points, assignments)
		entry := scores[cluster]
		entry.sum += coef
		entry.count++
		scores[cluster] = entry
	}

	result := make(map[int]float64)
	for cluster, entry := range scores {
		if entry.count > 0 {
			result[cluster] = entry.sum / float64(entry.count)
		}
	}
	return result
}

package vectors

import (
	"math"
	"math/rand"
	"slices"
)

const (
	maxKMeansIters    = 100
	convergenceTol    = 1e-6
	silhouetteSampleN = 1000
)

// KMeansResult holds the output of a K-means clustering run.
type KMeansResult struct {
	K           int
	Assignments []int       // Assignments[i] = cluster ID for points[i]
	Centroids   [][]float32 // Centroids[j] = centroid vector for cluster j
	Iterations  int
}

// KMeans runs K-means clustering with K-means++ initialization on the given
// points using cosine distance. Points must be non-empty, uniform-dimensional,
// and L2-normalized. Invalid point shapes return an empty result.
func KMeans(points [][]float32, k int, rng *rand.Rand) *KMeansResult {
	n := len(points)
	if n == 0 {
		return &KMeansResult{K: 0}
	}
	dims, ok := kmeansInputDims(points)
	if !ok {
		return &KMeansResult{K: 0}
	}
	if k > n {
		k = n
	}
	if k < 1 {
		k = 1
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(int64(n*k) + 1))
	}

	centroids := kmeansInit(points, k, rng)
	assignments := make([]int, n)

	var iters int
	for i := range maxKMeansIters {
		iters = i + 1
		changed := kmeansAssign(points, centroids, assignments)
		newCentroids := kmeansUpdate(points, k, dims, assignments)

		maxDrift := centroidDrift(centroids, newCentroids)
		centroids = newCentroids

		if !changed || maxDrift < convergenceTol {
			break
		}
	}

	return &KMeansResult{K: k, Assignments: assignments, Centroids: centroids, Iterations: iters}
}

func kmeansInputDims(points [][]float32) (int, bool) {
	dims := len(points[0])
	if dims == 0 {
		return 0, false
	}
	for _, p := range points {
		if len(p) != dims {
			return 0, false
		}
	}
	return dims, true
}

// kmeansInit selects k initial centroids via K-means++ (proportional to
// squared cosine distance from nearest existing centroid).
func kmeansInit(points [][]float32, k int, rng *rand.Rand) [][]float32 {
	n := len(points)
	centroids := make([][]float32, 0, k)
	centroids = append(centroids, slices.Clone(points[rng.Intn(n)]))

	// dists[i] = squared cosine distance from point i to its nearest centroid so far.
	// Initialized to +Inf so the first centroid update populates all entries.
	dists := make([]float32, n)
	for i := range dists {
		dists[i] = math.MaxFloat32
	}

	updateDists := func(c []float32) float64 {
		var total float64
		for i, p := range points {
			d := 1 - Cosine32(p, c)
			if sq := d * d; sq < dists[i] {
				dists[i] = sq
			}
			total += float64(dists[i])
		}
		return total
	}

	// Seed dists from the first centroid. total is kept across iterations so
	// we never need an extra scan to recompute it.
	total := updateDists(centroids[0])

	for range k - 1 {
		threshold := rng.Float64() * total
		var cumulative float64
		chosen := 0
		for i, d := range dists {
			cumulative += float64(d)
			if cumulative >= threshold {
				chosen = i
				break
			}
		}
		c := slices.Clone(points[chosen])
		centroids = append(centroids, c)
		total = updateDists(c)
	}
	return centroids
}

// kmeansAssign assigns each point to its nearest centroid by cosine distance.
// Returns true if any assignment changed.
func kmeansAssign(points, centroids [][]float32, assignments []int) bool {
	changed := false
	for i, p := range points {
		best := 0
		bestSim := float32(-2) // cosine range is [-1, 1]
		for j, c := range centroids {
			s := Cosine32(p, c)
			if s > bestSim {
				bestSim = s
				best = j
			}
		}
		if assignments[i] != best {
			assignments[i] = best
			changed = true
		}
	}
	return changed
}

// kmeansUpdate recomputes centroids as the mean of assigned points,
// then L2-normalizes (spherical K-means). Empty clusters are
// reinitialized from the point furthest from its assigned centroid.
func kmeansUpdate(points [][]float32, k, dims int, assignments []int) [][]float32 {
	centroids := make([][]float32, k)
	counts := make([]int, k)
	for j := range k {
		centroids[j] = make([]float32, dims)
	}

	for i, p := range points {
		c := assignments[i]
		counts[c]++
		for d, v := range p {
			centroids[c][d] += v
		}
	}

	for j := range k {
		if counts[j] == 0 {
			centroids[j] = recoverEmpty(points, centroids, assignments)
			continue
		}
		cnt := float32(counts[j])
		for d := range centroids[j] {
			centroids[j][d] /= cnt
		}
		Normalize32(centroids[j])
	}
	return centroids
}

// recoverEmpty finds the point with the greatest cosine distance from its
// assigned centroid and returns a clone of it as a replacement centroid.
func recoverEmpty(points, centroids [][]float32, assignments []int) []float32 {
	worstIdx := 0
	worstDist := float32(-1)
	for i, p := range points {
		d := 1 - Cosine32(p, centroids[assignments[i]])
		if d > worstDist {
			worstDist = d
			worstIdx = i
		}
	}
	return slices.Clone(points[worstIdx])
}

// centroidDrift returns the maximum cosine distance between old and new centroids.
func centroidDrift(old, cur [][]float32) float32 {
	var maxDrift float32
	for i := range old {
		d := 1 - Cosine32(old[i], cur[i])
		if d > maxDrift {
			maxDrift = d
		}
	}
	return maxDrift
}

// SilhouetteScore computes the average silhouette coefficient for the given
// clustering. For large datasets (n > silhouetteSampleN), a deterministic
// subsample is used to keep computation tractable.
func SilhouetteScore(points [][]float32, assignments []int, k int) float64 {
	n := len(points)
	if n < 2 || k < 2 {
		return 0
	}

	// Deterministic subsample for large datasets.
	indices := make([]int, n)
	for i := range n {
		indices[i] = i
	}
	if n > silhouetteSampleN {
		r := rand.New(rand.NewSource(int64(n * k)))
		r.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		indices = indices[:silhouetteSampleN]
	}

	// Hoist per-cluster accumulators out of the per-point loop to avoid
	// allocating two slices of size k for every sampled point.
	sums := make([]float64, k)
	counts := make([]int, k)

	var total float64
	for _, idx := range indices {
		total += pointSilhouette(idx, points, assignments, k, sums, counts)
	}
	return total / float64(len(indices))
}

// pointSilhouette computes the silhouette coefficient for a single point.
// sums and counts are caller-provided scratch buffers (len == k); they are
// zeroed on entry and may be reused across calls.
func pointSilhouette(idx int, points [][]float32, assignments []int, k int, sums []float64, counts []int) float64 {
	// Zero the scratch buffers.
	for i := range k {
		sums[i] = 0
		counts[i] = 0
	}

	myCluster := assignments[idx]

	// Accumulate distances per cluster.
	for j, p := range points {
		if j == idx {
			continue
		}
		d := float64(1 - Cosine32(points[idx], p))
		c := assignments[j]
		sums[c] += d
		counts[c]++
	}

	// a = mean intra-cluster distance.
	a := 0.0
	if counts[myCluster] > 0 {
		a = sums[myCluster] / float64(counts[myCluster])
	}

	// b = min mean distance to any other cluster.
	b := math.MaxFloat64
	for c := range k {
		if c == myCluster || counts[c] == 0 {
			continue
		}
		avg := sums[c] / float64(counts[c])
		if avg < b {
			b = avg
		}
	}
	if b == math.MaxFloat64 {
		return 0 // only one cluster has points
	}

	denom := max(a, b)
	if denom == 0 {
		return 0
	}
	return (b - a) / denom
}

// FindOptimalK runs K-means for each candidate k in [minK, maxK] and returns
// the k with the highest silhouette score.
func FindOptimalK(points [][]float32, minK, maxK int, rng *rand.Rand) (int, float64) {
	n := len(points)
	if _, ok := kmeansInputDims(points); !ok {
		return 0, 0
	}
	if minK < 1 {
		minK = 1
	}
	if maxK > n {
		maxK = n
	}
	if maxK < minK {
		return 0, 0
	}

	bestK := minK
	bestScore := -2.0

	for k := minK; k <= maxK; k++ {
		result := KMeans(points, k, rng)
		score := SilhouetteScore(points, result.Assignments, result.K)
		if score > bestScore {
			bestScore = score
			bestK = k
		}
	}
	return bestK, bestScore
}

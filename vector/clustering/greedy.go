package clustering

import (
	"cmp"
	"slices"

	"github.com/dotcommander/reliquary/vector"
)

// clusterGreedy performs greedy similarity-sorted clustering.
// This preserves the original behavior from iterative.go.
func (s *service) clusterGreedy(embeddings [][]float64, opts ClusterOptions) (*ClusterResult, error) {
	n := len(embeddings)
	if n == 0 {
		return &ClusterResult{}, nil
	}

	targetMax := opts.TargetMax
	if targetMax <= 0 {
		targetMax = 8
	}

	hardMax := opts.HardMax
	if hardMax <= 0 {
		hardMax = 12
	}

	maxSizeBytes := int64(opts.MaxSizeKB * 1024)
	if maxSizeBytes <= 0 {
		maxSizeBytes = 80 * 1024
	}

	// Track which embeddings have been assigned
	assigned := make([]bool, n)
	assignments := make([]int, n)
	for i := range assignments {
		assignments[i] = -1
	}

	var centroids [][]float64
	clusterID := 0

	for {
		// Find first unassigned embedding as seed
		seedIdx := -1
		for i := 0; i < n; i++ {
			if !assigned[i] {
				seedIdx = i
				break
			}
		}
		if seedIdx == -1 {
			break // all assigned
		}

		// Score all unassigned embeddings by similarity to seed
		type scored struct {
			idx   int
			score float64
			size  int64
		}
		var candidates []scored
		for i := 0; i < n; i++ {
			if assigned[i] || i == seedIdx {
				continue
			}
			sim := vectors.Cosine64(embeddings[seedIdx], embeddings[i])
			var size int64
			if opts.FileSizes != nil && i < len(opts.FileSizes) {
				size = opts.FileSizes[i]
			}
			candidates = append(candidates, scored{idx: i, score: sim, size: size})
		}

		// Sort by similarity descending
		slices.SortFunc(candidates, func(a, b scored) int { return cmp.Compare(b.score, a.score) })

		// Build cluster
		cluster := []int{seedIdx}
		var seedSize int64
		if opts.FileSizes != nil && seedIdx < len(opts.FileSizes) {
			seedSize = opts.FileSizes[seedIdx]
		}
		totalSize := seedSize

		for _, c := range candidates {
			if len(cluster) >= hardMax {
				break
			}

			// Skip if would exceed size limit
			if maxSizeBytes > 0 && totalSize+c.size > maxSizeBytes {
				continue
			}

			// Skip if below threshold
			if opts.Threshold > 0 && c.score < opts.Threshold {
				continue
			}

			cluster = append(cluster, c.idx)
			totalSize += c.size

			if len(cluster) >= targetMax {
				break
			}
		}

		// Assign cluster
		for _, idx := range cluster {
			assigned[idx] = true
			assignments[idx] = clusterID
		}

		// Compute centroid
		clusterEmbs := make([][]float64, len(cluster))
		for i, idx := range cluster {
			clusterEmbs[i] = embeddings[idx]
		}
		centroids = append(centroids, ComputeCentroid(clusterEmbs))

		clusterID++
	}

	silhouette := 0.0
	if clusterID > 1 {
		silhouette = AverageSilhouetteScore(embeddings, assignments)
	}

	return &ClusterResult{
		Assignments: assignments,
		K:           clusterID,
		Centroids:   centroids,
		Silhouette:  silhouette,
	}, nil
}

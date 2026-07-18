package clustering

import (
	"math"
)

// Linkage specifies the linkage method for HAC.
type Linkage string

const (
	LinkageSingle   Linkage = "single"   // minimum distance between clusters
	LinkageComplete Linkage = "complete" // maximum distance between clusters
	LinkageAverage  Linkage = "average"  // average distance between clusters
)

// HACConfig holds configuration for hierarchical agglomerative clustering.
type HACConfig struct {
	K       int     // target number of clusters (0 = auto via silhouette)
	Linkage Linkage // linkage method (default: average)
}

// DefaultHACConfig returns default HAC configuration.
func DefaultHACConfig() HACConfig {
	return HACConfig{
		K:       0,
		Linkage: LinkageAverage,
	}
}

// HACResult holds the result of HAC clustering.
type HACResult struct {
	Assignments []int       // cluster ID for each point
	Centroids   [][]float64 // cluster centroids
	K           int         // number of clusters
	Dendrogram  []MergeStep // merge history (for analysis)
}

// MergeStep records a merge in the dendrogram.
type MergeStep struct {
	ClusterA int     // first cluster merged
	ClusterB int     // second cluster merged
	Distance float64 // distance at merge
	NewSize  int     // size of merged cluster
}

// hacCluster represents a cluster during HAC.
type hacCluster struct {
	id       int
	members  []int     // indices of original points
	centroid []float64 // for computing distances
	active   bool      // whether cluster still exists
}

// HAC performs hierarchical agglomerative clustering.
// Uses cosine distance since embeddings are normalized.
func HAC(embeddings [][]float64, cfg HACConfig) *HACResult {
	n := len(embeddings)
	if n == 0 {
		return &HACResult{
			Assignments: nil,
			Centroids:   nil,
			K:           0,
		}
	}

	k := cfg.K
	if k <= 0 {
		k = 2 // minimum sensible default
	}
	if k > n {
		k = n
	}

	linkage := cfg.Linkage
	if linkage == "" {
		linkage = LinkageAverage
	}

	// Initialize each point as its own cluster
	clusters := make([]*hacCluster, n)
	for i, emb := range embeddings {
		centroid := make([]float64, len(emb))
		copy(centroid, emb)
		clusters[i] = &hacCluster{
			id:       i,
			members:  []int{i},
			centroid: centroid,
			active:   true,
		}
	}

	// Precompute distance matrix
	distMatrix := DistanceMatrix(embeddings, CosineDistance)

	// Merge history
	var dendrogram []MergeStep
	nextClusterID := n

	// Merge until we have k clusters
	numActive := n
	for numActive > k {
		// Find pair with minimum distance
		minDist := math.Inf(1)
		mergeI, mergeJ := -1, -1

		for i := 0; i < len(clusters); i++ {
			if !clusters[i].active {
				continue
			}
			for j := i + 1; j < len(clusters); j++ {
				if !clusters[j].active {
					continue
				}

				d := clusterDistance(clusters[i], clusters[j], distMatrix, linkage)
				if d < minDist {
					minDist = d
					mergeI, mergeJ = i, j
				}
			}
		}

		if mergeI == -1 {
			break // no more pairs to merge
		}

		// Merge clusters
		cA, cB := clusters[mergeI], clusters[mergeJ]

		// Record merge
		dendrogram = append(dendrogram, MergeStep{
			ClusterA: cA.id,
			ClusterB: cB.id,
			Distance: minDist,
			NewSize:  len(cA.members) + len(cB.members),
		})

		// Create merged cluster
		newMembers := make([]int, 0, len(cA.members)+len(cB.members))
		newMembers = append(newMembers, cA.members...)
		newMembers = append(newMembers, cB.members...)

		// Compute new centroid
		newCentroid := computeMergedCentroid(embeddings, newMembers)

		mergedCluster := &hacCluster{
			id:       nextClusterID,
			members:  newMembers,
			centroid: newCentroid,
			active:   true,
		}
		nextClusterID++

		// Deactivate merged clusters and add new one
		cA.active = false
		cB.active = false
		clusters = append(clusters, mergedCluster)
		numActive--
	}

	// Build final assignments
	assignments := make([]int, n)
	var centroids [][]float64

	clusterID := 0
	for _, c := range clusters {
		if !c.active {
			continue
		}
		for _, memberIdx := range c.members {
			assignments[memberIdx] = clusterID
		}
		centroids = append(centroids, c.centroid)
		clusterID++
	}

	return &HACResult{
		Assignments: assignments,
		Centroids:   centroids,
		K:           clusterID,
		Dendrogram:  dendrogram,
	}
}

// clusterDistance computes distance between two clusters using specified linkage.
func clusterDistance(a, b *hacCluster, distMatrix [][]float64, linkage Linkage) float64 {
	switch linkage {
	case LinkageSingle:
		// Minimum distance between any two points
		minDist := math.Inf(1)
		for _, i := range a.members {
			for _, j := range b.members {
				if distMatrix[i][j] < minDist {
					minDist = distMatrix[i][j]
				}
			}
		}
		return minDist

	case LinkageComplete:
		// Maximum distance between any two points
		maxDist := 0.0
		for _, i := range a.members {
			for _, j := range b.members {
				if distMatrix[i][j] > maxDist {
					maxDist = distMatrix[i][j]
				}
			}
		}
		return maxDist

	case LinkageAverage:
		// Average distance between all pairs
		return averageLinkage(a, b, distMatrix)

	default:
		// Default to average linkage
		return averageLinkage(a, b, distMatrix)
	}
}

// averageLinkage computes the average distance between all member pairs of two
// clusters, returning +Inf when either cluster has no members.
func averageLinkage(a, b *hacCluster, distMatrix [][]float64) float64 {
	sumDist := 0.0
	count := 0
	for _, i := range a.members {
		for _, j := range b.members {
			sumDist += distMatrix[i][j]
			count++
		}
	}
	if count == 0 {
		return math.Inf(1)
	}
	return sumDist / float64(count)
}

// computeMergedCentroid computes the centroid of the member sub-slice by
// delegating to ComputeCentroid (vectors.ComputeCentroid64), which provides the
// single-point fast path and ragged-input (nil) safety.
func computeMergedCentroid(embeddings [][]float64, members []int) []float64 {
	if len(members) == 0 || len(embeddings) == 0 {
		return nil
	}

	points := make([][]float64, len(members))
	for i, idx := range members {
		points[i] = embeddings[idx]
	}
	return ComputeCentroid(points)
}

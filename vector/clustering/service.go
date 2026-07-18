// Package clustering performs online and offline document clustering over
// caller-supplied [][]float64 embeddings. It never embeds anything itself, so
// the caller owns vector-space consistency across the whole pipeline.
//
// K-means and silhouette helpers delegate to the root vector
// package so vector math has one owner. This package keeps the service API,
// greedy size-constrained clustering, and hierarchical clustering orchestration.
//
// # Service
//
// NewClusterService selects one of three algorithms by name — "greedy"
// (similarity-based, size-constrained grouping), "kmeans" (spherical k-means
// with optional silhouette-based auto-k), or "hac" (hierarchical agglomerative
// clustering) — and returns a ClusterService whose Cluster method drives them
// through a common ClusterOptions/ClusterResult contract.
//
// # Direct helpers
//
// The underlying primitives are also exported for callers that want finer
// control: FindOptimalK and AverageSilhouetteScore for k-means and silhouette
// scoring, and HAC with CutDendrogram for hierarchical clustering.
package clustering

// ClusterService defines the interface for clustering embeddings.
type ClusterService interface {
	// Cluster groups embeddings into clusters.
	Cluster(embeddings [][]float64, opts ClusterOptions) (*ClusterResult, error)
}

// ClusterOptions configures the clustering behavior.
type ClusterOptions struct {
	Algorithm string // "greedy", "kmeans", "hac"
	K         int    // 0 = auto-select via silhouette
	MaxK      int    // max k for auto-selection (default: 20)
	Linkage   string // for HAC: "single", "complete", "average"

	// Greedy-specific options (for backward compatibility)
	TargetMax int     // target files per cluster (default: 8)
	HardMax   int     // hard cap on cluster size (default: 12)
	MaxSizeKB int     // max total size in KB (default: 80)
	FileSizes []int64 // file sizes in bytes (for greedy size constraints)
	Threshold float64 // minimum similarity threshold for greedy
}

// ClusterResult holds the output of clustering.
type ClusterResult struct {
	Assignments []int       // cluster ID for each embedding
	K           int         // number of clusters
	Centroids   [][]float64 // cluster centroids (for k-means)
	Silhouette  float64     // average silhouette score
}

// DefaultClusterOptions returns sensible defaults.
func DefaultClusterOptions() ClusterOptions {
	return ClusterOptions{
		Algorithm: "greedy",
		K:         0,
		MaxK:      20,
		Linkage:   "average",
		TargetMax: 8,
		HardMax:   12,
		MaxSizeKB: 80,
		Threshold: 0.0, // no minimum threshold
	}
}

// service implements ClusterService.
type service struct {
	algorithm string
}

// Cluster performs clustering based on the configured algorithm.
func (s *service) Cluster(embeddings [][]float64, opts ClusterOptions) (*ClusterResult, error) {
	if len(embeddings) == 0 {
		return &ClusterResult{
			Assignments: nil,
			K:           0,
			Centroids:   nil,
			Silhouette:  0,
		}, nil
	}

	algorithm := opts.Algorithm
	if algorithm == "" {
		algorithm = s.algorithm
	}

	switch algorithm {
	case "kmeans":
		return s.clusterKMeans(embeddings, opts)
	case "hac":
		return s.clusterHAC(embeddings, opts)
	default:
		return s.clusterGreedy(embeddings, opts)
	}
}

// clusterKMeans performs K-means clustering with optional auto-k via silhouette.
func (s *service) clusterKMeans(embeddings [][]float64, opts ClusterOptions) (*ClusterResult, error) {
	k := opts.K

	if k <= 0 {
		// Auto-select k via silhouette
		maxK := opts.MaxK
		if maxK <= 0 {
			maxK = 20
		}
		silResult := FindOptimalK(embeddings, SilhouetteConfig{
			MinK:      2,
			MaxK:      maxK,
			Algorithm: "kmeans",
		})
		return &ClusterResult{
			Assignments: silResult.Assignments,
			K:           silResult.BestK,
			Centroids:   silResult.Centroids,
			Silhouette:  silResult.BestScore,
		}, nil
	}

	// Fixed k
	result := KMeans(embeddings, KMeansConfig{
		K:             k,
		MaxIterations: 100,
		Tolerance:     1e-4,
	})

	silhouette := AverageSilhouetteScore(embeddings, result.Assignments)

	return &ClusterResult{
		Assignments: result.Assignments,
		K:           result.K,
		Centroids:   result.Centroids,
		Silhouette:  silhouette,
	}, nil
}

// clusterHAC performs hierarchical agglomerative clustering with optional auto-k.
func (s *service) clusterHAC(embeddings [][]float64, opts ClusterOptions) (*ClusterResult, error) {
	k := opts.K

	linkage := Linkage(opts.Linkage)
	if linkage == "" {
		linkage = LinkageAverage
	}

	if k <= 0 {
		// Auto-select k via silhouette
		maxK := opts.MaxK
		if maxK <= 0 {
			maxK = 20
		}
		silResult := FindOptimalK(embeddings, SilhouetteConfig{
			MinK:      2,
			MaxK:      maxK,
			Algorithm: "hac",
		})
		return &ClusterResult{
			Assignments: silResult.Assignments,
			K:           silResult.BestK,
			Centroids:   silResult.Centroids,
			Silhouette:  silResult.BestScore,
		}, nil
	}

	// Fixed k
	result := HAC(embeddings, HACConfig{
		K:       k,
		Linkage: linkage,
	})

	silhouette := AverageSilhouetteScore(embeddings, result.Assignments)

	return &ClusterResult{
		Assignments: result.Assignments,
		K:           result.K,
		Centroids:   result.Centroids,
		Silhouette:  silhouette,
	}, nil
}

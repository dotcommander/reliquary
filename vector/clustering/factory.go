package clustering

// NewClusterService creates a ClusterService for the given algorithm.
// Supported algorithms: "greedy" (default), "kmeans", "hac"
func NewClusterService(algorithm string) ClusterService {
	if algorithm == "" {
		algorithm = "greedy"
	}

	return &service{
		algorithm: algorithm,
	}
}

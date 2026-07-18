package clustering

// CutDendrogram cuts the dendrogram at a specific distance threshold.
// Returns cluster assignments.
func CutDendrogram(dendrogram []MergeStep, n int, distanceThreshold float64) []int {
	// Start with each point in its own cluster
	assignments := make([]int, n)
	for i := range assignments {
		assignments[i] = i
	}

	// Apply merges that happen below the threshold
	for _, step := range dendrogram {
		if step.Distance > distanceThreshold {
			break
		}
		// Find all points in cluster A and B, merge them
		// This requires tracking which cluster each point belongs to
		// For simplicity, we rebuild from assignments
		targetCluster := -1
		for i, a := range assignments {
			if a == step.ClusterA || a == step.ClusterB {
				if targetCluster == -1 {
					targetCluster = step.ClusterA
				}
				assignments[i] = targetCluster
			}
		}
	}

	// Renumber clusters contiguously
	return renumberAssignments(assignments)
}

// renumberAssignments renumbers assignments to be contiguous starting from 0.
func renumberAssignments(assignments []int) []int {
	mapping := make(map[int]int)
	nextID := 0

	result := make([]int, len(assignments))
	for i, a := range assignments {
		if _, exists := mapping[a]; !exists {
			mapping[a] = nextID
			nextID++
		}
		result[i] = mapping[a]
	}

	return result
}

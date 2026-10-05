package clustering

// CutDendrogram cuts the dendrogram at a specific distance threshold.
// Returns cluster assignments.
func CutDendrogram(dendrogram []MergeStep, n int, distanceThreshold float64) []int {
	// Representatives map both original leaves and HAC's synthetic IDs to
	// original leaves. Union-find preserves membership across chained merges.
	parents := make([]int, n)
	representatives := make([]int, n+len(dendrogram))
	for i := range parents {
		parents[i] = i
		representatives[i] = i
	}
	find := func(leaf int) int {
		for parents[leaf] != leaf {
			parents[leaf] = parents[parents[leaf]]
			leaf = parents[leaf]
		}
		return leaf
	}

	// Apply merges that happen below the threshold
	for merge, step := range dendrogram {
		if step.Distance > distanceThreshold {
			break
		}
		a := find(representatives[step.ClusterA])
		b := find(representatives[step.ClusterB])
		parents[b] = a
		representatives[n+merge] = a
	}
	assignments := make([]int, n)
	for leaf := range assignments {
		assignments[leaf] = find(leaf)
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

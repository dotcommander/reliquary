package clustering

import "testing"

func TestCutDendrogram(t *testing.T) {
	t.Parallel()

	dendrogram := []MergeStep{
		{ClusterA: 0, ClusterB: 1, Distance: 0.1},
		{ClusterA: 0, ClusterB: 2, Distance: 0.5},
		{ClusterA: 0, ClusterB: 3, Distance: 1.2},
	}

	// Cut at distance 0.3 -> only merge 0 and 1
	assignments := CutDendrogram(dendrogram, 4, 0.3)
	// Points 0, 1 in same cluster (ID 0), point 2 in ID 1, point 3 in ID 2
	if len(assignments) != 4 {
		t.Fatalf("assignments len = %d, want 4", len(assignments))
	}
	if assignments[0] != assignments[1] {
		t.Fatalf("points 0 and 1 should share cluster assignment: %v", assignments)
	}
	if assignments[2] == assignments[0] || assignments[3] == assignments[0] {
		t.Fatalf("points 2 and 3 should be in distinct clusters: %v", assignments)
	}

	// Cut at distance 0.6 -> merge 0, 1, 2
	assignments6 := CutDendrogram(dendrogram, 4, 0.6)
	if assignments6[0] != assignments6[1] || assignments6[0] != assignments6[2] {
		t.Fatalf("points 0, 1, 2 should share cluster: %v", assignments6)
	}
}

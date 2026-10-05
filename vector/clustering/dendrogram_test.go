package clustering

import (
	"math"
	"slices"
	"testing"
)

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

func TestCutDendrogramSyntheticChainedMerges(t *testing.T) {
	t.Parallel()
	dendrogram := []MergeStep{
		{ClusterA: 0, ClusterB: 1, Distance: 0.1, NewSize: 2}, // ID 5
		{ClusterA: 2, ClusterB: 3, Distance: 0.2, NewSize: 2}, // ID 6
		{ClusterA: 5, ClusterB: 6, Distance: 0.3, NewSize: 4}, // ID 7
		{ClusterA: 7, ClusterB: 4, Distance: 0.4, NewSize: 5},
	}
	for _, tc := range []struct {
		threshold float64
		want      []int
	}{
		{0, []int{0, 1, 2, 3, 4}},
		{0.2, []int{0, 0, 1, 1, 2}},
		{0.3, []int{0, 0, 0, 0, 1}},
		{0.4, []int{0, 0, 0, 0, 0}},
	} {
		got := CutDendrogram(dendrogram, 5, tc.threshold)
		if !slices.Equal(got, tc.want) {
			t.Fatalf("threshold=%v: got %v, want %v", tc.threshold, got, tc.want)
		}
	}
}

func TestCutDendrogramConsumesHACHistory(t *testing.T) {
	t.Parallel()
	points := [][]float64{{1, 0}, {1, 0.01}, {1, 0.02}, {0, 1}, {0.01, 1}}
	result := HAC(points, HACConfig{K: 1, Linkage: LinkageAverage})
	if got := CutDendrogram(result.Dendrogram, len(points), math.Inf(1)); !slices.Equal(got, []int{0, 0, 0, 0, 0}) {
		t.Fatalf("full HAC cut = %v, want all leaves in one cluster", got)
	}
}

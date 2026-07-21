package vectors

import (
	"math"
	"math/rand"
	"testing"
)

func TestKMeans_TwoClusters(t *testing.T) {
	t.Parallel()
	const dims = 8
	rng := rand.New(rand.NewSource(42))

	points := make([][]float32, 20)
	for i := range 10 {
		v := make([]float32, dims)
		v[0] = 1.0
		v[1] = rng.Float32() * 0.1
		Normalize32(v)
		points[i] = v
	}
	for i := range 10 {
		v := make([]float32, dims)
		v[4] = 1.0
		v[5] = rng.Float32() * 0.1
		Normalize32(v)
		points[10+i] = v
	}

	result := KMeans(points, 2, rand.New(rand.NewSource(99)))
	if result.K != 2 {
		t.Fatalf("K = %d, want 2", result.K)
	}
	if len(result.Assignments) != 20 {
		t.Fatalf("assignments len = %d, want 20", len(result.Assignments))
	}

	clusterA := result.Assignments[0]
	for i := range 10 {
		if result.Assignments[i] != clusterA {
			t.Fatalf("assignment[%d] = %d, want clusterA %d", i, result.Assignments[i], clusterA)
		}
	}
	clusterB := result.Assignments[10]
	if clusterA == clusterB {
		t.Fatalf("clusterA == clusterB == %d, want distinct clusters", clusterA)
	}
	for i := 10; i < 20; i++ {
		if result.Assignments[i] != clusterB {
			t.Fatalf("assignment[%d] = %d, want clusterB %d", i, result.Assignments[i], clusterB)
		}
	}
}

func TestKMeans_InvalidInputsDoNotPanic(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		points [][]float32
		k      int
		rng    *rand.Rand
	}{
		{
			name:   "nil rng uses deterministic fallback",
			points: [][]float32{{1, 0}, {0, 1}},
			k:      2,
			rng:    nil,
		},
		{
			name:   "ragged longer row returns empty result",
			points: [][]float32{{1, 0}, {0, 1, 0}},
			k:      2,
			rng:    rand.New(rand.NewSource(1)),
		},
		{
			name:   "ragged shorter row returns empty result",
			points: [][]float32{{1, 0}, {0}},
			k:      2,
			rng:    rand.New(rand.NewSource(1)),
		},
		{
			name:   "zero dimension returns empty result",
			points: [][]float32{{}, {}},
			k:      2,
			rng:    rand.New(rand.NewSource(1)),
		},
		{
			name:   "NaN returns empty result",
			points: [][]float32{{1, 0}, {float32(math.NaN()), 1}},
			k:      2,
			rng:    rand.New(rand.NewSource(1)),
		},
		{
			name:   "infinity returns empty result",
			points: [][]float32{{1, 0}, {0, float32(math.Inf(1))}},
			k:      2,
			rng:    rand.New(rand.NewSource(1)),
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			result := KMeans(tc.points, tc.k, tc.rng)
			if result == nil {
				t.Fatal("KMeans returned nil result")
			}
			if tc.rng == nil {
				if result.K != 2 || len(result.Assignments) != len(tc.points) {
					t.Fatalf("nil rng result = %+v, want usable clustering", result)
				}
				return
			}
			if result.K != 0 || len(result.Assignments) != 0 || len(result.Centroids) != 0 {
				t.Fatalf("invalid input result = %+v, want empty result", result)
			}
		})
	}
}

func TestFindOptimalK_InvalidInputs(t *testing.T) {
	t.Parallel()

	if k, score := FindOptimalK([][]float32{{1, 0}, {0, 1, 0}}, 1, 3, nil); k != 0 || score != 0 {
		t.Fatalf("ragged FindOptimalK = (%d, %v), want (0, 0)", k, score)
	}
	if k, score := FindOptimalK([][]float32{{1, 0}, {0, 1}}, 3, 1, nil); k != 0 || score != 0 {
		t.Fatalf("inverted bounds FindOptimalK = (%d, %v), want (0, 0)", k, score)
	}
	if k, score := FindOptimalK([][]float32{{1, 0}, {0, 1}}, 0, 5, nil); k < 1 || k > 2 || math.IsNaN(score) {
		t.Fatalf("clamped FindOptimalK = (%d, %v), want k in [1,2] and non-NaN score", k, score)
	}
}

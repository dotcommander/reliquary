package vectors

import (
	"math"
	"reflect"
	"testing"
)

// deterministicVecs builds n L2-normalized dim-dimensional vectors from a fixed
// seed via a simple LCG — no math/rand, fully reproducible across runs.
func deterministicVecs(n, dim int, seed uint64) [][]float32 {
	state := seed
	next := func() float32 {
		state = state*6364136223846793005 + 1442695040888963407
		// map high bits to [-1, 1)
		return float32(int64(state>>33))/float32(1<<30) - 1
	}
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for d := range v {
			v[d] = next()
		}
		Normalize32(v)
		out[i] = v
	}
	return out
}

func bruteForcePairs(vecs [][]float32, t float32) map[[2]int]struct{} {
	want := map[[2]int]struct{}{}
	for i := range vecs {
		if len(vecs[i]) == 0 {
			continue
		}
		for j := i + 1; j < len(vecs); j++ {
			if len(vecs[j]) == 0 {
				continue
			}
			if Cosine32(vecs[i], vecs[j]) >= t {
				want[[2]int{i, j}] = struct{}{}
			}
		}
	}
	return want
}

func TestNearDuplicateGroups_ParallelPairAndOrthogonalSingleton(t *testing.T) {
	t.Parallel()

	// Two near-parallel vectors (cosine ~0.95) plus an orthogonal one.
	a := []float32{1, 0, 0}
	// b tilted slightly off a: cosine = 1/sqrt(1+e^2); pick e for ~0.95.
	e := float32(math.Sqrt(1.0/(0.95*0.95) - 1.0))
	b := []float32{1, e, 0}
	c := []float32{0, 0, 1} // orthogonal to both

	vecs := [][]float32{a, b, c}
	Normalize32(vecs[0])
	Normalize32(vecs[1])
	Normalize32(vecs[2])

	if got := Cosine32(vecs[0], vecs[1]); got < 0.94 {
		t.Fatalf("cosine = %v, want >= 0.94", got)
	}

	groups := NearDuplicateGroups(vecs, 0.9)
	if len(groups) != 1 {
		t.Fatalf("groups len = %d, want exactly one group", len(groups))
	}
	if !reflect.DeepEqual(groups[0], []int{0, 1}) {
		t.Fatalf("groups[0] = %#v, want [0 1]", groups[0])
	}
}

func TestNearDuplicatePairs_RecallMatchesBruteForce(t *testing.T) {
	t.Parallel()

	vecs := deterministicVecs(40, 16, 0x9E3779B97F4A7C15)

	for _, thr := range []float32{0.3, 0.5, 0.7, 0.9} {
		thr := thr
		got := NearDuplicatePairs(vecs, thr)
		gotSet := map[[2]int]struct{}{}
		for _, p := range got {
			gotSet[p] = struct{}{}
		}
		want := bruteForcePairs(vecs, thr)

		if len(gotSet) != len(want) {
			t.Fatalf("pair count at threshold %v = %d, want %d", thr, len(gotSet), len(want))
		}
		for p := range want {
			_, ok := gotSet[p]
			if !ok {
				t.Fatalf("prefilter dropped true positive %v at threshold %v", p, thr)
			}
		}
		for p := range gotSet {
			_, ok := want[p]
			if !ok {
				t.Fatalf("prefilter produced false positive %v at threshold %v", p, thr)
			}
		}
	}
}

func TestNearDuplicateGroups_ThresholdMonotonicity(t *testing.T) {
	t.Parallel()

	vecs := deterministicVecs(30, 16, 0xDEADBEEFCAFEBABE)

	// Membership map for a given threshold: index -> set of co-members.
	coMembers := func(thr float32) map[int]map[int]struct{} {
		m := map[int]map[int]struct{}{}
		for _, g := range NearDuplicateGroups(vecs, thr) {
			for _, i := range g {
				if m[i] == nil {
					m[i] = map[int]struct{}{}
				}
				for _, j := range g {
					if i != j {
						m[i][j] = struct{}{}
					}
				}
			}
		}
		return m
	}

	low := coMembers(0.5)
	high := coMembers(0.7)

	// Raising the threshold must never ADD a co-membership.
	for i, hs := range high {
		for j := range hs {
			_, ok := low[i][j]
			if !ok {
				t.Fatalf("raising threshold grew a group: %d-%d present at 0.7 but not 0.5", i, j)
			}
		}
	}
}

func TestNearDuplicate_DegenerateInputs(t *testing.T) {
	t.Parallel()

	if got := NearDuplicateGroups(nil, 0.9); len(got) != 0 {
		t.Fatalf("NearDuplicateGroups(nil) = %#v, want empty", got)
	}
	if got := NearDuplicatePairs(nil, 0.9); len(got) != 0 {
		t.Fatalf("NearDuplicatePairs(nil) = %#v, want empty", got)
	}

	if got := NearDuplicateGroups([][]float32{}, 0.9); len(got) != 0 {
		t.Fatalf("NearDuplicateGroups(empty) = %#v, want empty", got)
	}
	if got := NearDuplicatePairs([][]float32{}, 0.9); len(got) != 0 {
		t.Fatalf("NearDuplicatePairs(empty) = %#v, want empty", got)
	}

	single := [][]float32{{1, 0, 0}}
	if got := NearDuplicateGroups(single, 0.9); len(got) != 0 {
		t.Fatalf("NearDuplicateGroups(single) = %#v, want empty", got)
	}
	if got := NearDuplicatePairs(single, 0.9); len(got) != 0 {
		t.Fatalf("NearDuplicatePairs(single) = %#v, want empty", got)
	}

	// nil / zero-length members are skipped, never linked.
	withNils := [][]float32{{1, 0, 0}, nil, {}, {1, 0, 0}}
	groups := NearDuplicateGroups(withNils, 0.99)
	if len(groups) != 1 {
		t.Fatalf("groups len = %d, want 1", len(groups))
	}
	if !reflect.DeepEqual(groups[0], []int{0, 3}) {
		t.Fatalf("groups[0] = %#v, want [0 3]", groups[0])
	}

	pairs := NearDuplicatePairs(withNils, 0.99)
	if len(pairs) != 1 {
		t.Fatalf("pairs len = %d, want 1", len(pairs))
	}
	if pairs[0] != [2]int{0, 3} {
		t.Fatalf("pairs[0] = %#v, want [0 3]", pairs[0])
	}
}

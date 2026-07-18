package vectors

import (
	"cmp"
	"slices"
)

// Near-duplicate detection by embedding cosine similarity.
//
// This is the SEMANTIC complement to the lexical SimHash near-dup primitive in
// the `dedup` module: SimHash catches near-identical surface text, while these
// functions catch vectors that are close in embedding space regardless of
// wording.
//
// Prefilter path: v1 verifies every pair with exact Cosine32 (a
// brute-force-correct O(n^2) baseline). Binary quantization (ComputeMedians +
// Quantize) is computed as the cheap O(n) screen a future radius-bounded
// prefilter would consume, but it does NOT reject pairs in v1: median-threshold
// quantization admits no safe, tight Hamming<->cosine radius bound, so pruning
// by Hamming radius could drop true positives. Until a provably-safe bound is
// established, correctness wins — the result is exactly the brute-force cosine
// set, so the prefilter cannot drop a true positive. The binary screen is
// wired in for the optimization but currently non-rejecting.

// NearDuplicateGroups groups input vectors into clusters of mutual near-duplicates
// by cosine similarity. Two vectors are linked when their cosine >= threshold;
// groups are the connected components of that graph (singletons omitted).
// Returns indices into vecs. Uses binary quantization as an O(n) Hamming
// prefilter, then verifies candidate pairs with exact Cosine32.
//
// Guards: nil/empty input or fewer than 2 usable vectors returns an empty
// result; nil or zero-length member vectors are skipped (never linked).
func NearDuplicateGroups(vecs [][]float32, cosineThreshold float32) [][]int {
	pairs := NearDuplicatePairs(vecs, cosineThreshold)
	if len(pairs) == 0 {
		return [][]int{}
	}

	uf := newUnionFind(len(vecs))
	for _, p := range pairs {
		uf.union(p[0], p[1])
	}

	// Collect members per connected-component root, preserving ascending index
	// order within each group for deterministic output.
	members := make(map[int][]int)
	for _, p := range pairs {
		for _, idx := range []int{p[0], p[1]} {
			root := uf.find(idx)
			members[root] = append(members[root], idx)
		}
	}

	groups := make([][]int, 0, len(members))
	for _, raw := range members {
		seen := make(map[int]struct{}, len(raw))
		uniq := make([]int, 0, len(raw))
		for _, idx := range raw {
			if _, ok := seen[idx]; ok {
				continue
			}
			seen[idx] = struct{}{}
			uniq = append(uniq, idx)
		}
		if len(uniq) < 2 {
			continue
		}
		slices.Sort(uniq)
		groups = append(groups, uniq)
	}

	// Deterministic group order: by first (smallest) member index.
	slices.SortFunc(groups, func(a, b []int) int {
		return cmp.Compare(a[0], b[0])
	})
	return groups
}

// NearDuplicatePairs returns the linked index pairs (i<j, cosine>=threshold)
// rather than connected-component groups. Pairs are returned in ascending
// (i, j) order. Same guards as NearDuplicateGroups.
func NearDuplicatePairs(vecs [][]float32, cosineThreshold float32) [][2]int {
	out := [][2]int{}
	if len(vecs) < 2 {
		return out
	}

	usable := make([]bool, len(vecs))
	usableVecs := make([][]float32, 0, len(vecs))
	for i, v := range vecs {
		if len(v) > 0 {
			usable[i] = true
			usableVecs = append(usableVecs, v)
		}
	}
	if len(usableVecs) < 2 {
		return out
	}

	// v1 verify step: exact Cosine32 over every usable pair. The binary screen
	// (computed below) is documented machinery and a future pruning hook, but it
	// does NOT reject pairs here — median-threshold quantization admits no safe,
	// tight Hamming<->cosine radius bound, so pruning by Hamming radius could
	// drop true positives. Correctness over cleverness: the result equals a
	// brute-force O(n^2) Cosine32 scan. (Quantization is still exercised so the
	// optimization can be turned on once a provably-safe bound is established.)
	_ = buildHammingScreen(vecs, usable)

	for i := 0; i < len(vecs); i++ {
		if !usable[i] {
			continue
		}
		for j := i + 1; j < len(vecs); j++ {
			if !usable[j] {
				continue
			}
			if Cosine32(vecs[i], vecs[j]) >= cosineThreshold {
				out = append(out, [2]int{i, j})
			}
		}
	}
	return out
}

// buildHammingScreen quantizes every usable vector against per-dimension medians,
// returning the binary vectors (nil entry for vectors that could not be
// quantized). Returns nil when quantization is unavailable (ragged dims). This
// is the O(n) screen that a future radius-bounded prefilter would consume; v1
// computes it but verifies all pairs exactly (see NearDuplicatePairs).
func buildHammingScreen(vecs [][]float32, usable []bool) []BinaryVector {
	medians := ComputeMedians(filterUsable(vecs, usable))
	if len(medians) == 0 {
		return nil
	}

	bins := make([]BinaryVector, len(vecs))
	dim := len(medians)
	for i, v := range vecs {
		if !usable[i] || len(v) != dim {
			continue
		}
		bv, err := Quantize(v, medians)
		if err != nil {
			continue
		}
		bins[i] = bv
	}
	return bins
}

func filterUsable(vecs [][]float32, usable []bool) [][]float32 {
	out := make([][]float32, 0, len(vecs))
	for i, v := range vecs {
		if usable[i] {
			out = append(out, v)
		}
	}
	return out
}

// --- small internal union-find over vector indices ---

type unionFind struct {
	parent []int
	rank   []int
}

func newUnionFind(n int) *unionFind {
	uf := &unionFind{parent: make([]int, n), rank: make([]int, n)}
	for i := range uf.parent {
		uf.parent[i] = i
	}
	return uf
}

func (uf *unionFind) find(x int) int {
	for uf.parent[x] != x {
		uf.parent[x] = uf.parent[uf.parent[x]] // path halving
		x = uf.parent[x]
	}
	return x
}

func (uf *unionFind) union(a, b int) {
	ra, rb := uf.find(a), uf.find(b)
	if ra == rb {
		return
	}
	if uf.rank[ra] < uf.rank[rb] {
		ra, rb = rb, ra
	}
	uf.parent[rb] = ra
	if uf.rank[ra] == uf.rank[rb] {
		uf.rank[ra]++
	}
}

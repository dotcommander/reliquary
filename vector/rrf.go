package vectors

import "slices"

// Scored pairs an index with its fused score.
// Sorted by descending score, then ascending index for deterministic tie-breaks.
type Scored struct {
	Index int
	Score float64
}

// defaultRRFK is the smoothing constant from Cormack et al. (2009).
// Larger values reduce the dominance of top ranks.
const defaultRRFK = 60

// RRF fuses ranked index lists via Reciprocal Rank Fusion.
// Each list is a ranking where position i has rank i+1.
//
// Each index contribution is:
//
//	score += 1/(k+rank)
//
// where rank is 1-based.
//
// Raw metric scores are not used; only rank positions contribute.
//
// Returns the fused ranking sorted by descending score and ascending index for ties,
// and the maximum score (or 0 for empty input).
func RRF(ranked [][]int, k float64) ([]Scored, float64) {
	if k <= 0 {
		k = defaultRRFK
	}

	acc := make(map[int]float64)
	for _, list := range ranked {
		for i, idx := range list {
			rank := float64(i + 1)
			acc[idx] += 1.0 / (k + rank)
		}
	}

	out := make([]Scored, 0, len(acc))
	var max float64
	for idx, score := range acc {
		out = append(out, Scored{Index: idx, Score: score})
		if score > max {
			max = score
		}
	}

	slices.SortFunc(out, func(a, b Scored) int {
		if a.Score != b.Score {
			if a.Score < b.Score {
				return 1
			}
			return -1
		}
		return a.Index - b.Index
	})

	return out, max
}

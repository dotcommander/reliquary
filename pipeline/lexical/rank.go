package lexical

import (
	"cmp"
	"slices"
)

// ScoreSpace identifies the semantics of Result.Score.
//
// Scores are comparable only within the same scoring identity. Database FTS
// scores, local BM25 scores, rank-only adapters, and fused RRF scores should not
// be mixed as raw numeric values.
type ScoreSpace string

const (
	// ScoreSpaceUnspecified means the caller has not declared score semantics.
	ScoreSpaceUnspecified ScoreSpace = ""
	// ScoreSpaceLocalBM25 marks package-local BM25 scores from BM25Score.
	ScoreSpaceLocalBM25 ScoreSpace = "local_bm25"
	// ScoreSpaceProvider marks provider-local scores such as ts_rank or
	// app-normalized SQLite bm25() values.
	ScoreSpaceProvider ScoreSpace = "provider"
	// ScoreSpaceRankOnly marks external results adapted by rank order only.
	ScoreSpaceRankOnly ScoreSpace = "rank_only"
	// ScoreSpaceRRF marks reciprocal-rank-fusion output scores.
	ScoreSpaceRRF ScoreSpace = "rrf"
)

// Result is one provider-neutral lexical result.
type Result struct {
	ID         string
	Score      float64
	Rank       int
	Source     string
	ScoreSpace ScoreSpace
	Metadata   map[string]string
}

// Candidate is kept as an alias for callers that model local candidates before
// deciding whether they came from BM25, FTS, or fused rank order.
type Candidate = Result

// RankedList is ordered by descending score, then ascending ID for ties when
// produced through Sort or RankByScore.
type RankedList []Candidate

// Sort orders results by descending score and ascending ID.
func (list RankedList) Sort() {
	slices.SortFunc(list, func(a, b Candidate) int {
		return compareCandidates(a, b)
	})
}

// RankByScore copies and sorts results by descending score and ascending ID.
func RankByScore(results []Candidate) RankedList {
	out := append(RankedList{}, results...)
	assignMissingScoreSpace(out, ScoreSpaceProvider)
	out.Sort()
	assignRanks(out)
	if out == nil {
		return RankedList{}
	}
	return out
}

// FromRanks converts an externally ranked ID list into a rank-only RankedList.
//
// Use this for DB-backed or external systems where raw scores are provider-local
// and only rank order is portable across fusion/evaluation layers.
func FromRanks(ids []string, source string) RankedList {
	results := make([]Result, 0, len(ids))
	for _, id := range ids {
		results = append(results, Result{ID: id, Source: source})
	}
	return RankByOrder(results, source, ScoreSpaceRankOnly)
}

// RankByOrder adapts already-ordered provider results without treating raw
// provider scores as globally comparable. Empty IDs are skipped.
func RankByOrder(results []Result, source string, scoreSpace ScoreSpace) RankedList {
	if scoreSpace == ScoreSpaceUnspecified {
		scoreSpace = ScoreSpaceRankOnly
	}
	out := make(RankedList, 0, len(results))
	for _, result := range results {
		if result.ID == "" {
			continue
		}
		if result.Source == "" {
			result.Source = source
		}
		if result.ScoreSpace == ScoreSpaceUnspecified {
			result.ScoreSpace = scoreSpace
		}
		out = append(out, result)
	}
	for i := range out {
		out[i].Rank = i + 1
		if out[i].ScoreSpace == ScoreSpaceRankOnly {
			out[i].Score = 1.0 / float64(out[i].Rank)
		}
	}
	if out == nil {
		return RankedList{}
	}
	return out
}

// RankByProviderRank sorts provider results by ascending positive Rank. Results
// without a positive Rank keep their relative input order after ranked results.
func RankByProviderRank(results []Result, source string, scoreSpace ScoreSpace) RankedList {
	type positioned struct {
		result Result
		index  int
	}
	items := make([]positioned, 0, len(results))
	for i, result := range results {
		if result.ID == "" {
			continue
		}
		items = append(items, positioned{result: result, index: i})
	}
	slices.SortStableFunc(items, func(a, b positioned) int {
		aRank := a.result.Rank
		bRank := b.result.Rank
		switch {
		case aRank > 0 && bRank > 0:
			return cmp.Compare(aRank, bRank)
		case aRank > 0:
			return -1
		case bRank > 0:
			return 1
		default:
			return cmp.Compare(a.index, b.index)
		}
	})

	ordered := make([]Result, len(items))
	for i, item := range items {
		ordered[i] = item.result
	}
	return RankByOrder(ordered, source, scoreSpace)
}

// FusionInput is one already-ranked source list for rank fusion.
type FusionInput struct {
	Source     string
	Candidates RankedList
	Weight     float64
}

// FusionOptions configures reciprocal-rank fusion.
type FusionOptions struct {
	// K is the RRF rank constant. Values <= 0 use 60.
	K float64
	// Limit caps returned candidates. Values <= 0 return all candidates.
	Limit int
}

// FuseRRFByID fuses ranked sources by stable document ID using reciprocal rank
// fusion. Duplicate IDs within one input list contribute only their first rank.
func FuseRRFByID(inputs []FusionInput, options FusionOptions) RankedList {
	k := options.K
	if k <= 0 {
		k = 60
	}

	scores := make(map[string]float64)
	sources := make(map[string]string)
	for _, input := range inputs {
		weight := input.Weight
		if weight == 0 {
			weight = 1
		}
		seen := make(map[string]struct{}, len(input.Candidates))
		for rank, candidate := range input.Candidates {
			if candidate.ID == "" {
				continue
			}
			if _, ok := seen[candidate.ID]; ok {
				continue
			}
			seen[candidate.ID] = struct{}{}
			scores[candidate.ID] += weight / (k + float64(rank+1))
			if sources[candidate.ID] == "" {
				sources[candidate.ID] = candidate.Source
				if sources[candidate.ID] == "" {
					sources[candidate.ID] = input.Source
				}
			}
		}
	}

	fused := make(RankedList, 0, len(scores))
	for id, score := range scores {
		fused = append(fused, Candidate{ID: id, Score: score, Source: sources[id], ScoreSpace: ScoreSpaceRRF})
	}
	fused.Sort()
	if options.Limit > 0 && options.Limit < len(fused) {
		fused = fused[:options.Limit]
	}
	assignRanks(fused)
	if fused == nil {
		return RankedList{}
	}
	return fused
}

// RankedIDs returns IDs in rank order.
func RankedIDs(list RankedList) []string {
	ids := make([]string, len(list))
	for i, item := range list {
		ids[i] = item.ID
	}
	return ids
}

func assignMissingScoreSpace(list RankedList, scoreSpace ScoreSpace) {
	for i := range list {
		if list[i].ScoreSpace == ScoreSpaceUnspecified {
			list[i].ScoreSpace = scoreSpace
		}
	}
}

func assignRanks(list RankedList) {
	for i := range list {
		list[i].Rank = i + 1
	}
}

func compareCandidates(a, b Candidate) int {
	if a.Score != b.Score {
		if a.Score < b.Score {
			return 1
		}
		return -1
	}
	if a.ID < b.ID {
		return -1
	}
	if a.ID > b.ID {
		return 1
	}
	if a.Source < b.Source {
		return -1
	}
	if a.Source > b.Source {
		return 1
	}
	return 0
}

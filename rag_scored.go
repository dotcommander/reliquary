package reliquary

import (
	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func scoreRawCandidates(weights retrieval.Weights, query string, candidates []indexcontract.ScoredCandidate, explain, sortResults bool) ([]*retrieval.Result, explanationMap) {
	raw := make([]retrieval.RawCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.Result == nil {
			continue
		}
		raw = append(raw, retrieval.RawCandidate{Result: candidate.Result, Raw: candidate.Raw, Present: candidate.Present})
	}
	ranked, traces := retrieval.NewScorer(weights).RecalibrateRaw(query, raw, explain, sortResults)
	if !explain {
		return ranked, nil
	}
	explanations := make(explanationMap, len(ranked))
	for n, result := range ranked {
		explanations[result] = &retrieval.SearchExplanation{Hybrid: traces[n], HybridRank: n + 1, HybridScoreUsed: sortResults}
	}
	return ranked, explanations
}

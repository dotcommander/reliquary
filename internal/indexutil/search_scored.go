package indexutil

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary/embedding"
	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

// SearchScoredCandidates selects against the full filtered pool, retaining
// uncalibrated signals beside independently owned selected payloads.
func SearchScoredCandidates(ctx context.Context, query indexcontract.IndexQuery, items []*retrieval.Result) ([]indexcontract.ScoredCandidate, error) {
	if err := ValidateFilter(query.Filter); err != nil {
		return nil, fmt.Errorf("reliquary index: %w", err)
	}
	candidates := make([]*retrieval.Result, 0, len(items))
	for _, item := range items {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if item == nil || !MatchesFilter(item, query.Filter) {
			continue
		}
		if len(query.Vector) > 0 && len(item.Embedding) > 0 && len(query.Vector) != len(item.Embedding) {
			return nil, fmt.Errorf("%w: query has %d dimensions, item %q has %d", indexcontract.ErrDimensionMismatch, len(query.Vector), item.ID, len(item.Embedding))
		}
		cp := *item
		cp.Explain = nil
		candidates = append(candidates, &cp)
	}
	selected := retrieval.NewScorer(retrieval.DefaultWeights()).SelectWithRaw(retrieval.EmbeddingVector(embedding.Vector(query.Vector)), query.Text, candidates, query.Limit)
	if len(selected) == 0 {
		return nil, nil
	}
	output := make([]indexcontract.ScoredCandidate, len(selected))
	for n, candidate := range selected {
		if query.RawText != "" {
			retrieval.CompleteLaneSignals(&candidate, nil, query.RawText, true)
		}
		if len(query.RawVector) > 0 {
			retrieval.CompleteLaneSignals(&candidate, retrieval.EmbeddingVector(embedding.Vector(query.RawVector)), "", false)
		}
		output[n] = indexcontract.ScoredCandidate{Result: clone(candidate.Result), Raw: candidate.Raw, Present: candidate.Present}
	}
	return output, nil
}

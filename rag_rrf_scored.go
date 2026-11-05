package reliquary

import (
	"context"
	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func (a *App) searchRRFLane(ctx context.Context, query IndexQuery) ([]*retrieval.Result, map[string]indexcontract.ScoredCandidate, error) {
	if searcher, ok := a.index.(indexcontract.ScoredCandidateSearcher); ok {
		candidates, err := searcher.SearchScoredCandidates(ctx, query)
		if err != nil {
			return nil, nil, err
		}
		items := make([]*retrieval.Result, len(candidates))
		byID := make(map[string]indexcontract.ScoredCandidate, len(candidates))
		for n, candidate := range candidates {
			items[n] = candidate.Result
			if candidate.Result != nil {
				byID[candidate.Result.ID] = candidate
			}
		}
		return items, byID, nil
	}
	query.RawText = ""
	query.RawVector = nil
	items, err := a.index.Search(ctx, query)
	return items, nil, err
}

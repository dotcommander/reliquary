package inmem

import (
	"context"
	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/internal/indexutil"
)

// SearchResultsOwned asserts that searches return detached payloads.
func (*Index) SearchResultsOwned() {}

// SearchScoredCandidates retrieves selected candidates and their raw signals.
func (i *Index) SearchScoredCandidates(ctx context.Context, query indexcontract.IndexQuery) ([]indexcontract.ScoredCandidate, error) {
	items, err := i.searchItems(ctx, query)
	if err != nil {
		return nil, err
	}
	return indexutil.SearchScoredCandidates(ctx, query, items)
}

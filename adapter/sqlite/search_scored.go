package sqlite

import (
	"context"
	"fmt"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/internal/indexutil"
)

// SearchScoredCandidates shares the legacy SQL candidate boundary while
// retaining raw signals from the single indexutil selection pass.
func (i *Index) SearchScoredCandidates(ctx context.Context, query indexcontract.IndexQuery) ([]indexcontract.ScoredCandidate, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := indexutil.ValidateFilter(query.Filter); err != nil {
		return nil, fmt.Errorf("sqlite index: %w", err)
	}
	items, err := i.searchItems(ctx, query)
	if err != nil {
		return nil, err
	}
	return indexutil.SearchScoredCandidates(ctx, query, items)
}

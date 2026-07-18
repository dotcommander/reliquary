package indexutil

import (
	"cmp"
	"context"
	"fmt"
	"reflect"
	"slices"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

// Search copies, filters, scores, and limits an in-memory candidate set.
func Search(ctx context.Context, query indexcontract.IndexQuery, items []*retrieval.Result) ([]*retrieval.Result, error) {
	candidates := make([]*retrieval.Result, 0, len(items))
	for _, item := range items {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if item == nil || !matches(item, query.Filter) {
			continue
		}
		if len(query.Vector) > 0 && len(item.Embedding) > 0 && len(query.Vector) != len(item.Embedding) {
			return nil, fmt.Errorf("%w: query has %d dimensions, item %q has %d", indexcontract.ErrDimensionMismatch, len(query.Vector), item.ID, len(item.Embedding))
		}
		cp := clone(item)
		candidates = append(candidates, cp)
	}
	if len(candidates) == 0 {
		return nil, nil
	}
	ranked := retrieval.NewScorer(retrieval.DefaultWeights()).RerankEmbedding(query.Vector, query.Text, candidates)
	// Scorer deliberately ranks only by score. Add an ID tie-break so database
	// and in-memory implementations expose the same deterministic contract.
	slices.SortStableFunc(ranked, func(a, b *retrieval.Result) int {
		if order := cmp.Compare(b.CombinedScore, a.CombinedScore); order != 0 {
			return order
		}
		return cmp.Compare(a.ID, b.ID)
	})
	if query.Limit > 0 && query.Limit < len(ranked) {
		ranked = ranked[:query.Limit]
	}
	return ranked, nil
}

// Clone returns a result whose embedding and top-level metadata map do not
// alias stored state. Metadata values retain their original value semantics.
func Clone(item *retrieval.Result) *retrieval.Result { return clone(item) }

func clone(item *retrieval.Result) *retrieval.Result {
	cp := *item
	cp.Embedding = slices.Clone(item.Embedding)
	if item.Metadata != nil {
		cp.Metadata = make(map[string]any, len(item.Metadata))
		for key, value := range item.Metadata {
			cp.Metadata[key] = value
		}
	}
	return &cp
}

func matches(item *retrieval.Result, filter map[string]any) bool {
	for key, want := range filter {
		var got any
		switch key {
		case "id":
			got = item.ID
		case "document_id":
			got = item.DocumentID
		case "filename":
			got = item.Filename
		default:
			got = item.Metadata[key]
		}
		if !reflect.DeepEqual(got, want) {
			return false
		}
	}
	return true
}

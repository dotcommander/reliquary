package indexutil

import (
	"context"
	"fmt"
	"slices"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

// Search copies, filters, scores, and limits an in-memory candidate set.
func Search(ctx context.Context, query indexcontract.IndexQuery, items []*retrieval.Result) ([]*retrieval.Result, error) {
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
		// Scoring mutates scalar fields only; payloads stay read-only until
		// selection so discarded candidates need no deep ownership copy.
		cp := *item
		cp.Explain = nil
		candidates = append(candidates, &cp)
	}
	if len(candidates) == 0 {
		return nil, nil
	}
	// Calibrate against every candidate before selecting the limited score/ID prefix.
	ranked := retrieval.NewScorer(retrieval.DefaultWeights()).RerankTopByID(retrieval.EmbeddingVector(query.Vector), query.Text, candidates, query.Limit)
	selected := make([]*retrieval.Result, len(ranked))
	for n, item := range ranked {
		selected[n] = clone(item)
	}
	return selected, nil
}

// Clone returns a result whose embedding and canonical JSON metadata containers
// do not alias stored state. Nested map[string]any objects and []any arrays are
// cloned recursively. Typed containers such as map[string]string retain their
// original value semantics and are outside this isolation guarantee.
func Clone(item *retrieval.Result) *retrieval.Result { return clone(item) }

func clone(item *retrieval.Result) *retrieval.Result {
	cp := *item
	cp.Explain = nil
	cp.Embedding = slices.Clone(item.Embedding)
	if item.Metadata != nil {
		cp.Metadata = cloneJSONObject(item.Metadata)
	}
	return &cp
}

func cloneJSONValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		if value == nil {
			return map[string]any(nil)
		}
		return cloneJSONObject(value)
	case []any:
		if value == nil {
			return []any(nil)
		}
		cloned := make([]any, len(value))
		for i, item := range value {
			cloned[i] = cloneJSONValue(item)
		}
		return cloned
	default:
		return value
	}
}

func cloneJSONObject(object map[string]any) map[string]any {
	cloned := make(map[string]any, len(object))
	for key, value := range object {
		cloned[key] = cloneJSONValue(value)
	}
	return cloned
}

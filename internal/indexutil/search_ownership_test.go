package indexutil

import (
	"context"
	"reflect"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestLimitedSearchPreservesScoresAndPayloadIsolation(t *testing.T) {
	t.Parallel()
	items := []*retrieval.Result{
		{ID: "b", Content: "alpha", Embedding: []float64{1, 0},
			Metadata:      map[string]any{"nested": []any{"original"}},
			CombinedScore: 99, Explain: &retrieval.SearchExplanation{FinalRank: 99}},
		{ID: "a", Content: "alpha", Embedding: []float64{1, 0}},
		{ID: "c", Content: "bravo", Embedding: []float64{0, 1}},
	}
	before := make([]*retrieval.Result, len(items))
	for i, item := range items {
		before[i] = Clone(item)
		before[i].Explain = item.Explain
	}
	query := indexcontract.IndexQuery{Text: "alpha", Vector: []float32{1, 0}}
	all, err := Search(context.Background(), query, items)
	if err != nil {
		t.Fatal(err)
	}
	query.Limit = 2
	limited, err := Search(context.Background(), query, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(limited) != 2 || limited[0].ID != "a" || limited[1].ID != "b" {
		t.Fatalf("limited order = %v, want a then b", limited)
	}
	if !reflect.DeepEqual(limited, all[:2]) {
		t.Fatalf("limit changed scores or payloads: got %+v, want %+v", limited, all[:2])
	}
	for _, result := range limited {
		if result.Explain != nil {
			t.Fatal("stored explanation leaked into search")
		}
		result.Embedding[0] = 9
		if result.Metadata != nil {
			result.Metadata["nested"].([]any)[0] = "changed"
		}
		result.CombinedScore = -1
	}
	if !reflect.DeepEqual(items, before) {
		t.Fatal("search or returned payload mutation changed source results")
	}
}

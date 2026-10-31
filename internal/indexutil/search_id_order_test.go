package indexutil

import (
	"context"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestSearchLimitRetainsFullPoolCalibration(t *testing.T) {
	t.Parallel()

	items := []*retrieval.Result{
		{ID: "b", Content: "alpha", Embedding: []float64{1, 0}},
		{ID: "a", Content: "alpha", Embedding: []float64{1, 0}},
		{ID: "outlier", Content: "bravo", Embedding: []float64{0, 1}},
	}
	query := indexcontract.IndexQuery{Text: "alpha", Vector: []float32{1, 0}}
	full, err := Search(context.Background(), query, items)
	if err != nil {
		t.Fatal(err)
	}
	query.Limit = 1
	limited, err := Search(context.Background(), query, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(limited) != 1 || limited[0].ID != "a" || limited[0].CombinedScore != full[0].CombinedScore || limited[0].EmbeddingScore != full[0].EmbeddingScore || limited[0].KeywordScore != full[0].KeywordScore {
		t.Fatalf("limited rank = %+v, want full-pool top %+v", limited, full[0])
	}
}

func TestSearchScoreThenIDOrderBeforeLimit(t *testing.T) {
	t.Parallel()

	items := []*retrieval.Result{
		{ID: "z", Content: "bravo", Embedding: []float64{0, 1}},
		{ID: "b", Content: "alpha", Embedding: []float64{1, 0}},
		{ID: "a", Content: "alpha", Embedding: []float64{1, 0}},
		{ID: "c", Content: "alpha", Embedding: []float64{1, 0}},
	}
	query := indexcontract.IndexQuery{Text: "alpha", Vector: []float32{1, 0}, Limit: 2}
	ranked, err := Search(context.Background(), query, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(ranked) != 2 || ranked[0].ID != "a" || ranked[1].ID != "b" {
		t.Fatalf("limited order = %v, want a then b", ranked)
	}
	if ranked[0].CombinedScore != ranked[1].CombinedScore || ranked[0].CombinedScore <= 0 {
		t.Fatalf("top tie scores = %v, %v, want equal positive scores", ranked[0].CombinedScore, ranked[1].CombinedScore)
	}
	query.Limit = 0
	all, err := Search(context.Background(), query, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 4 || all[0].ID != "a" || all[1].ID != "b" || all[2].ID != "c" || all[3].ID != "z" || all[2].CombinedScore <= all[3].CombinedScore {
		t.Fatalf("score/ID order = %v, want a, b, c, z", all)
	}
	for _, item := range items {
		if item.CombinedScore != 0 || item.EmbeddingScore != 0 || item.KeywordScore != 0 {
			t.Fatalf("source scores changed for %q: %+v", item.ID, item)
		}
	}
}

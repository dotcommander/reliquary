package reliquary

import (
	"context"
	"reflect"
	"testing"

	"github.com/dotcommander/reliquary/embed"
	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/index/inmem"
	"github.com/dotcommander/reliquary/internal/indexutil"
	"github.com/dotcommander/reliquary/retrieval"
)

type legacySearchIndex struct{ indexcontract.Index }

type aliasingSearchIndex struct {
	indexcontract.Index
	stored *retrieval.Result
}

func (i *aliasingSearchIndex) Search(context.Context, indexcontract.IndexQuery) ([]*retrieval.Result, error) {
	return []*retrieval.Result{i.stored}, nil
}

func TestSearchScoredDifferentialMatrix(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	idx := inmem.New()
	corpus := []*retrieval.Result{
		{ID: "a", DocumentID: "a", IndexIdentity: "space", Content: "alpha beta", Filename: "alpha.txt", Embedding: []float64{1, 0}, RecencyScore: 0.2},
		{ID: "b", DocumentID: "b", IndexIdentity: "space", Content: "beta gamma", Filename: "beta.txt", Embedding: []float64{0.8, 0.2}, ImportanceScore: 0.8},
		{ID: "c", DocumentID: "c", IndexIdentity: "space", Content: "alpha gamma", Filename: "gamma.txt", Embedding: []float64{0, 1}},
		{ID: "d", DocumentID: "d", IndexIdentity: "space", Content: "unrelated", Filename: "delta.txt", Embedding: []float64{-1, 0}},
	}
	if err := idx.Upsert(ctx, corpus); err != nil {
		t.Fatal(err)
	}
	for _, weights := range []struct {
		name  string
		value retrieval.Weights
	}{{"adaptive", retrieval.DefaultWeights()}, {"custom", retrieval.Weights{Embedding: 0.2, Keyword: 0.5, Filename: 0.3}}, {"salience", retrieval.Weights{Embedding: 0.63, Keyword: 0.27, Filename: 0.10, Recency: 0.1, Importance: 0.2}}} {
		for _, limit := range []int{0, 2, 4} {
			for _, rrf := range []bool{false, true} {
				for _, explain := range []bool{false, true} {
					name := weights.name
					if rrf {
						name += "/rrf"
					}
					if explain {
						name += "/explain"
					}
					if limit == 0 {
						name += "/unbounded"
					} else if limit == 2 {
						name += "/prefix"
					} else {
						name += "/full"
					}
					t.Run(name, func(t *testing.T) {
						t.Parallel()
						fresh, err := New(WithEmbedder(embed.NewHashing(2)), WithIndex(idx), WithIndexIdentity("space"), WithWeights(weights.value))
						if err != nil {
							t.Fatal(err)
						}
						legacy, err := New(WithEmbedder(embed.NewHashing(2)), WithIndex(&legacySearchIndex{idx}), WithIndexIdentity("space"), WithWeights(weights.value))
						if err != nil {
							t.Fatal(err)
						}
						opts := []SearchOption{CandidateLimit(limit)}
						if rrf {
							opts = append(opts, WithRRF(60))
						}
						if explain {
							opts = append(opts, WithExplain())
						}
						got, err := fresh.Search(ctx, "alpha beta", opts...)
						if err != nil {
							t.Fatal(err)
						}
						want, err := legacy.Search(ctx, "alpha beta", opts...)
						if err != nil {
							t.Fatal(err)
						}
						if !reflect.DeepEqual(got, want) {
							t.Fatalf("new != legacy:\n%#v\n%#v", got, want)
						}
					})
				}
			}
		}
	}
}

func TestSearchResultMutationPreservesStoredState(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	owned := inmem.New()
	original := &retrieval.Result{ID: "a", IndexIdentity: "space", DocumentID: "doc", Content: "alpha", Filename: "alpha.txt", Embedding: []float64{1, 0}, Metadata: map[string]any{"nested": map[string]any{"values": []any{"original"}}}, RecencyScore: 0.3, ImportanceScore: 0.4}
	if err := owned.Upsert(ctx, []*retrieval.Result{original}); err != nil {
		t.Fatal(err)
	}
	alias := &aliasingSearchIndex{stored: indexutil.Clone(original)}
	for _, tc := range []struct {
		name  string
		index indexcontract.Index
	}{{"owned", owned}, {"aliasing fallback", alias}} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			app, err := New(WithEmbedder(embed.NewHashing(2)), WithIndex(tc.index), WithIndexIdentity("space"))
			if err != nil {
				t.Fatal(err)
			}
			for _, rrf := range []bool{false, true} {
				opts := []SearchOption{WithExplain()}
				if rrf {
					opts = append(opts, WithRRF(60))
				}
				first, err := app.Search(ctx, "alpha", opts...)
				if err != nil || len(first) != 1 {
					t.Fatalf("first: %v %v", first, err)
				}
				item := first[0]
				item.ID = "mutated"
				item.IndexIdentity = "mutated"
				item.DocumentID = "mutated"
				item.Content = "mutated"
				item.Filename = "mutated"
				item.Embedding[0] = 9
				item.Embedding[1] = 9
				item.Metadata["nested"].(map[string]any)["values"].([]any)[0] = "mutated"
				item.EmbeddingScore = 9
				item.KeywordScore = 9
				item.FilenameScore = 9
				item.RecencyScore = 9
				item.ImportanceScore = 9
				item.CombinedScore = 9
				second, err := app.Search(ctx, "alpha", opts...)
				if err != nil || len(second) != 1 {
					t.Fatalf("second: %v %v", second, err)
				}
				got := second[0]
				if got.ID != "a" || got.IndexIdentity != "space" || got.DocumentID != "doc" || got.Content != "alpha" || got.Filename != "alpha.txt" || got.Embedding[0] != 1 || got.Embedding[1] != 0 || got.Metadata["nested"].(map[string]any)["values"].([]any)[0] != "original" || got.RecencyScore != 0.3 || got.ImportanceScore != 0.4 {
					t.Fatalf("stored state changed: %+v", got)
				}
			}
		})
	}
}

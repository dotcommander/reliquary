package retrieval_test

import (
	"context"
	"math"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestScorerTokenEmptyQueriesHaveNoLexicalSignal(t *testing.T) {
	t.Parallel()
	for _, query := range []string{"", " \t\n", "!!! ... --", "Go AI", "to be", "the and"} {
		t.Run(query, func(t *testing.T) {
			t.Parallel()
			scorer := retrieval.NewScorerWithOptions(retrieval.Weights{Embedding: 0.7, Keyword: 0.2, Filename: 0.1}, false)
			results := []*retrieval.Result{
				{ID: "aligned", Content: "alpha", Filename: "alpha.txt", Embedding: []float64{1, 0}},
				{ID: "orthogonal", Content: "beta", Filename: "beta.txt", Embedding: []float64{0, 1}},
			}
			ranked, traces := scorer.RerankWithTrace([]float64{1, 0}, query, results)
			for i, trace := range traces {
				if trace.Present.Keyword || trace.Present.Filename || !trace.Present.Embedding {
					t.Fatalf("presence = %#v, want embedding only", trace.Present)
				}
				if trace.Calibrated.Keyword != 0 || trace.Calibrated.Filename != 0 {
					t.Fatalf("absent channel calibrated to %#v", trace.Calibrated)
				}
				want := 0.7
				if i == 1 {
					want = 0
				}
				if math.Abs(ranked[i].CombinedScore-want) > 1e-12 || math.IsNaN(ranked[i].CombinedScore) {
					t.Fatalf("score = %v, want %v", ranked[i].CombinedScore, want)
				}
			}
			// Direct scoring must clear stale lexical values as well.
			item := &retrieval.Result{Content: "alpha", Filename: "alpha.txt", KeywordScore: 1, FilenameScore: 1}
			if got := scorer.Score(nil, query, item); got != 0 || item.KeywordScore != 0 || item.FilenameScore != 0 {
				t.Fatalf("Score token-empty query = %v, item = %#v", got, item)
			}
		})
	}
}

func TestScorerUsableSignalsKeepWeightingAndZeroOverlap(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		query               string
		wantTop, wantBottom float64
	}{
		{query: "alpha", wantTop: 1, wantBottom: 0},
		{query: "gamma", wantTop: 0.85, wantBottom: 0.15},
		// "not" is a usable token in this scorer, not a configured stopword.
		{query: "not", wantTop: 0.85, wantBottom: 0.15},
	} {
		t.Run(tc.query, func(t *testing.T) {
			t.Parallel()
			scorer := retrieval.NewScorerWithOptions(retrieval.Weights{Embedding: 0.7, Keyword: 0.2, Filename: 0.1}, false)
			ranked, traces := scorer.RerankWithTrace([]float64{1, 0}, tc.query, []*retrieval.Result{
				{ID: "aligned", Content: "alpha", Filename: "alpha.txt", Embedding: []float64{1, 0}},
				{ID: "orthogonal", Content: "beta", Filename: "beta.txt", Embedding: []float64{0, 1}},
			})
			for i, trace := range traces {
				if !trace.Present.Keyword || !trace.Present.Filename || !trace.Present.Embedding {
					t.Fatalf("usable channels absent: %#v", trace.Present)
				}
				want := tc.wantTop
				if i == 1 {
					want = tc.wantBottom
				}
				if math.IsNaN(ranked[i].CombinedScore) || math.IsInf(ranked[i].CombinedScore, 0) || math.Abs(ranked[i].CombinedScore-want) > 1e-12 {
					t.Fatalf("score = %v, want %v", ranked[i].CombinedScore, want)
				}
				if tc.query != "alpha" && (trace.Raw.Keyword != 0 || trace.Raw.Filename != 0 || trace.Calibrated.Keyword != 0.5 || trace.Calibrated.Filename != 0.5) {
					t.Fatalf("zero overlap lost real lexical presence: %#v", trace)
				}
			}
		})
	}
}

func TestScorerTokenEmptyCandidateChannelsRemainAbsent(t *testing.T) {
	t.Parallel()
	scorer := retrieval.NewScorerWithOptions(retrieval.Weights{Keyword: 0.8, Filename: 0.2}, false)
	_, traces := scorer.RerankWithTrace(nil, "alpha", []*retrieval.Result{
		{ID: "punctuation", Content: "!!!", Filename: "---"},
		{ID: "stopwords", Content: "to be", Filename: "Go AI"},
	})
	for _, trace := range traces {
		if trace.Present.Keyword || trace.Present.Filename || trace.CombinedScore != 0 {
			t.Fatalf("token-empty candidate has lexical signal: %#v", trace)
		}
	}
}

func TestFacadeTokenEmptyQueryHasNoCalibratedLexicalSignal(t *testing.T) {
	t.Parallel()
	app := reliquary.InMemory(32)
	ctx := context.Background()
	if _, err := app.Ingest(ctx,
		document.Document{ID: "alpha", Title: "alpha.txt", Text: "alpha reference"},
		document.Document{ID: "beta", Title: "beta.txt", Text: "beta reference"},
	); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{"!!!", "Go AI", "to be"} {
		results, err := app.Search(ctx, query, reliquary.WithExplain())
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 2 {
			t.Fatalf("query %q returned %d candidates, want 2", query, len(results))
		}
		for _, result := range results {
			if result.Explain == nil {
				t.Fatal("missing search explanation")
			}
			trace := result.Explain.Hybrid
			if trace.Present.Keyword || trace.Present.Filename || trace.Calibrated.Keyword != 0 || trace.Calibrated.Filename != 0 {
				t.Fatalf("query %q has false lexical signal: %#v", query, trace)
			}
			if math.IsNaN(result.CombinedScore) || math.IsInf(result.CombinedScore, 0) {
				t.Fatalf("query %q score = %v", query, result.CombinedScore)
			}
		}
	}
}

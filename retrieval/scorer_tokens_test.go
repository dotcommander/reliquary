package retrieval

import (
	"math"
	"testing"
)

func TestScorerPreparedQueryPreservesTokenSemantics(t *testing.T) {
	t.Parallel()
	query := "THE ALPHA alpha βήτα 42"
	scorer := NewScorer(DefaultWeights())
	items := []*Result{
		{ID: "alpha", Content: "alpha", Filename: "alpha.txt"},
		{ID: "beta", Content: "βήτα", Filename: "βήτα.txt"},
	}
	ranked, traces := scorer.RerankWithTrace(nil, query, items)
	for i, result := range ranked {
		wantKeyword := 1.0 / 3
		if result.ID == "alpha" {
			wantKeyword = 2.0 / 3
		}
		trace := traces[i]
		if trace.QueryTokenCount != 3 || trace.Raw.Keyword != wantKeyword || trace.Raw.Filename != 0.5 {
			t.Fatalf("unexpected raw signals for %s: %+v", result.ID, trace)
		}
		if trace.Weights != DefaultWeights() {
			t.Fatalf("weights = %+v, want defaults for three tokens", trace.Weights)
		}
		single := &Result{Content: result.Content, Filename: result.Filename}
		got := scorer.Score(nil, query, single)
		want := 0.27*wantKeyword + 0.10*0.5
		if math.Abs(got-want) > 1e-12 || single.KeywordScore != trace.Raw.Keyword || single.FilenameScore != trace.Raw.Filename {
			t.Fatalf("Score and rerank raw signals diverged for %s: %+v", result.ID, single)
		}
	}
}

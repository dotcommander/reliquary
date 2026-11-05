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

func TestOverlapRatioScanMatchesMapImplementation(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		tokens    []string
		reference []string
	}{
		{"empty probe", nil, []string{"alpha"}},
		{"empty reference", []string{"alpha"}, nil},
		{"both empty", nil, nil},
		{"no matches", []string{"alpha", "beta"}, []string{"gamma", "delta"}},
		{"all matches", []string{"alpha", "beta"}, []string{"beta", "alpha"}},
		{"duplicate probes counted", []string{"alpha", "alpha", "beta"}, []string{"alpha"}},
		{"duplicate reference harmless", []string{"alpha"}, []string{"alpha", "alpha", "alpha"}},
		{"large reference", []string{"alpha", "zulu"}, makeTokens("x", 64)[:63]},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got, want := overlapRatioScan(tc.tokens, tc.reference), overlapRatio(tc.tokens, tc.reference); got != want {
				t.Fatalf("overlapRatioScan = %v, want overlapRatio %v", got, want)
			}
		})
	}
}

func makeTokens(prefix string, n int) []string {
	tokens := make([]string, n)
	for i := range tokens {
		if i == 0 {
			tokens[i] = "alpha"
		} else {
			tokens[i] = prefix + string(rune('a'+i%26)) + string(rune('0'+i%10))
		}
	}
	return tokens
}

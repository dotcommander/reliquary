package lexical

import (
	"math"
	"reflect"
	"testing"
)

func TestBM25LengthNormalizationParameters(t *testing.T) {
	t.Parallel()
	short := NewDocumentStats([]string{"alpha"})
	long := NewDocumentStats([]string{"alpha", "beta", "gamma", "delta"})
	corpus := NewCorpusStats([]DocumentStats{short, long})
	query := NormalizeQuery("alpha", nil)
	for _, tc := range []struct {
		name   string
		params BM25Params
		wantB  float64
	}{
		{name: "whole-zero defaults", params: BM25Params{}, wantB: 0.75},
		{name: "explicit zero B", params: BM25Params{K1: 1.2, B: 0}, wantB: 0},
		{name: "valid B", params: BM25Params{K1: 1.2, B: 0.4}, wantB: 0.4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			for _, doc := range []DocumentStats{short, long} {
				idf := math.Log(1 + 0.5/2.5)
				want := idf * 2.2 / (1 + 1.2*(1-tc.wantB+tc.wantB*float64(doc.Length)/2.5))
				got := BM25Score(query, doc, corpus, tc.params)
				if math.Abs(got-want) > 1e-12 {
					t.Fatalf("length %d score = %.15f, want %.15f", doc.Length, got, want)
				}
			}
		})
	}
}

func TestRRFNormalizesInvalidKAndPreservesValidK(t *testing.T) {
	t.Parallel()
	inputs := []FusionInput{
		{Source: "left", Candidates: RankedList{{ID: "b"}, {ID: "a"}}},
		{Source: "right", Candidates: RankedList{{ID: "a"}, {ID: "b"}}},
	}
	for _, tc := range []struct {
		name  string
		k     float64
		wantK float64
	}{
		{"zero", 0, 60}, {"negative", -1, 60}, {"NaN", math.NaN(), 60},
		{"positive infinity", math.Inf(1), 60}, {"negative infinity", math.Inf(-1), 60},
		{"positive", 12, 12}, {"positive fraction", 0.5, 0.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := FuseRRFByID(inputs, FusionOptions{K: tc.k})
			if !reflect.DeepEqual(RankedIDs(got), []string{"a", "b"}) {
				t.Fatalf("tied IDs = %v, want [a b]", RankedIDs(got))
			}
			wantScore := 1/(tc.wantK+1) + 1/(tc.wantK+2)
			for i, candidate := range got {
				if math.IsNaN(candidate.Score) || math.IsInf(candidate.Score, 0) || math.Abs(candidate.Score-wantScore) > 1e-12 || candidate.Rank != i+1 {
					t.Fatalf("candidate = %#v, want finite score %v and rank %d", candidate, wantScore, i+1)
				}
			}
			if again := FuseRRFByID(inputs, FusionOptions{K: tc.k}); !reflect.DeepEqual(got, again) {
				t.Fatalf("rank changed across identical inputs: %v then %v", got, again)
			}
		})
	}
}

package retrieval

import (
	"math"
	"testing"

	"github.com/dotcommander/reliquary/vector"
)

// referenceMMR retains the prior full selected-set scan for parity checks.
func referenceMMR[T any](items []T, k int, lambda float64, relevance func(T) float64, similarity func(T, []T) float64, trace bool) ([]T, []MMRExplanation) {
	if k <= 0 || len(items) == 0 {
		return nil, nil
	}
	lambda = clamp01(lambda)
	remaining := append([]T(nil), items...)
	selected := make([]T, 0, min(k, len(items)))
	var traces []MMRExplanation
	if trace {
		traces = make([]MMRExplanation, 0, cap(selected))
	}
	for len(remaining) > 0 && len(selected) < k {
		bestIdx := 0
		bestScore := math.Inf(-1)
		bestSimilarity := 0.0
		hasBest := false
		for i, item := range remaining {
			itemSimilarity := similarity(item, selected)
			score := lambda*relevance(item) - (1-lambda)*itemSimilarity
			if i == 0 {
				bestSimilarity = itemSimilarity
			}
			if math.IsNaN(score) {
				continue
			}
			if !hasBest || score > bestScore {
				hasBest = true
				bestScore = score
				bestIdx = i
				bestSimilarity = itemSimilarity
			}
		}
		chosen := remaining[bestIdx]
		selected = append(selected, chosen)
		if trace {
			relevanceScore := relevance(chosen)
			relevanceContribution := lambda * relevanceScore
			penalty := -(1 - lambda) * bestSimilarity
			traces = append(traces, MMRExplanation{
				Lambda:                lambda,
				Relevance:             relevanceScore,
				MaxSimilarity:         bestSimilarity,
				RelevanceContribution: relevanceContribution,
				Penalty:               penalty,
				SelectionScore:        relevanceContribution + penalty,
			})
		}
		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
	}
	return selected, traces
}

func TestMMRMatchesPriorSelection(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		items  []MMRItem
		k      int
		lambda float64
	}{
		{name: "empty", k: 3, lambda: 0.5},
		{name: "zero k", items: []MMRItem{{ID: "a"}}, lambda: 0.5},
		{name: "negative k", items: []MMRItem{{ID: "a"}}, k: -1},
		{name: "ties and missing vectors", items: []MMRItem{
			{ID: "a", Score: 0.5}, {ID: "b", Score: 0.5, Embedding: []float64{1, 0}},
			{ID: "c", Score: 0.5, Embedding: []float64{1, 0}}, {ID: "d", Score: 0.5},
		}, k: 7, lambda: 0.5},
		{name: "max rises on later selection", items: []MMRItem{
			{ID: "a", Score: 1, Embedding: []float64{1, 0}},
			{ID: "b", Score: 0.9, Embedding: []float64{0, 1}},
			{ID: "c", Score: 0.8, Embedding: []float64{0, 1}},
			{ID: "d", Score: 0.7, Embedding: []float64{-1, 0}},
		}, k: 4, lambda: 0.7},
		{name: "opposite zero and mismatched", items: []MMRItem{
			{ID: "a", Score: 1, Embedding: []float64{1, 0}},
			{ID: "b", Score: 0.7, Embedding: []float64{-1, 0}},
			{ID: "c", Score: 0.8, Embedding: []float64{0, 0}},
			{ID: "d", Score: 0.6, Embedding: []float64{1}},
		}, k: 4, lambda: 0.4},
		{name: "NaN scores and vectors", items: []MMRItem{
			{ID: "nan", Score: math.NaN(), Embedding: []float64{1, 0}},
			{ID: "vector nan", Score: 0.7, Embedding: []float64{math.NaN(), 0}},
			{ID: "finite", Score: -2, Embedding: []float64{1, 0}},
			{ID: "minus infinity", Score: math.Inf(-1), Embedding: []float64{0, 1}},
		}, k: 4, lambda: 0.5},
		{name: "all NaN", items: []MMRItem{
			{ID: "a", Score: math.NaN(), Embedding: []float64{1, 0}},
			{ID: "b", Score: math.NaN(), Embedding: []float64{1, 0}},
			{ID: "c", Score: math.NaN(), Embedding: []float64{0, 1}},
		}, k: 3, lambda: math.NaN()},
		{name: "lambda below zero", items: []MMRItem{{ID: "a", Score: -1}, {ID: "b", Score: 2}}, k: 2, lambda: -2},
		{name: "lambda above one", items: []MMRItem{{ID: "a", Score: -1}, {ID: "b", Score: 2}}, k: 2, lambda: 2},
		{name: "infinite relevance", items: []MMRItem{
			{ID: "a", Score: math.Inf(1)}, {ID: "b", Score: math.Inf(-1)}, {ID: "c", Score: -3},
		}, k: 3, lambda: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			want, _ := referenceMMR(tc.items, tc.k, tc.lambda, func(item MMRItem) float64 {
				return item.Score
			}, maxSimilarity, false)
			got := MMR(tc.items, tc.k, tc.lambda)
			if (got == nil) != (want == nil) || len(got) != len(want) {
				t.Fatalf("MMR length = %d (nil %t), want %d (nil %t)", len(got), got == nil, len(want), want == nil)
			}
			for i := range want {
				if got[i].ID != want[i].ID {
					t.Fatalf("MMR rank %d = %q, want %q", i, got[i].ID, want[i].ID)
				}
			}

			results := make([]*Result, 0, len(tc.items)+2)
			results = append(results, nil)
			for _, item := range tc.items {
				results = append(results, &Result{ID: item.ID, CombinedScore: item.Score, Embedding: item.Embedding})
			}
			results = append(results, nil)
			candidates := results[1 : len(results)-1]
			wantResults, wantTraces := referenceMMR(candidates, tc.k, tc.lambda, func(result *Result) float64 {
				return result.CombinedScore
			}, func(item *Result, selected []*Result) float64 {
				maxSim := 0.0
				if len(item.Embedding) == 0 {
					return 0
				}
				for _, existing := range selected {
					if len(existing.Embedding) == 0 {
						continue
					}
					if sim := vectors.Cosine64(item.Embedding, existing.Embedding); sim > maxSim {
						maxSim = sim
					}
				}
				return maxSim
			}, true)
			gotResults, gotTraces := DiversifyWithTrace(results, tc.k, tc.lambda)
			plain := Diversify(results, tc.k, tc.lambda)
			if (gotResults == nil) != (wantResults == nil) || len(gotResults) != len(wantResults) || len(plain) != len(wantResults) || (gotTraces == nil) != (wantTraces == nil) || len(gotTraces) != len(wantTraces) {
				t.Fatalf("Diversify lengths: traced=%d plain=%d traces=%d; want results=%d traces=%d", len(gotResults), len(plain), len(gotTraces), len(wantResults), len(wantTraces))
			}
			for i := range wantResults {
				if gotResults[i] != wantResults[i] || plain[i] != wantResults[i] {
					t.Fatalf("Diversify rank %d = traced %p plain %p, want %p", i, gotResults[i], plain[i], wantResults[i])
				}
				gotFields := [...]float64{gotTraces[i].Lambda, gotTraces[i].Relevance, gotTraces[i].MaxSimilarity, gotTraces[i].RelevanceContribution, gotTraces[i].Penalty, gotTraces[i].SelectionScore}
				wantFields := [...]float64{wantTraces[i].Lambda, wantTraces[i].Relevance, wantTraces[i].MaxSimilarity, wantTraces[i].RelevanceContribution, wantTraces[i].Penalty, wantTraces[i].SelectionScore}
				for field := range gotFields {
					if gotFields[field] != wantFields[field] && !(math.IsNaN(gotFields[field]) && math.IsNaN(wantFields[field])) {
						t.Fatalf("trace rank %d field %d = %v, want %v", i, field, gotFields[field], wantFields[field])
					}
				}
			}
		})
	}
}

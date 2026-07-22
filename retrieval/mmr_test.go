package retrieval

import (
	"math"
	"testing"
)

func TestMMRAndDiversify(t *testing.T) {
	t.Parallel()

	r1 := &Result{ID: "r1", CombinedScore: 0.9, Embedding: []float64{1, 0}, Filename: "f1"}
	r2 := &Result{ID: "r2", CombinedScore: 0.85, Embedding: []float64{0.99, 0.01}, Filename: "f1"}
	r3 := &Result{ID: "r3", CombinedScore: 0.7, Embedding: []float64{0, 1}, Filename: "f2"}

	results := []*Result{nil, r1, r2, r3}

	t.Run("MMRItems filters nil", func(t *testing.T) {
		items := MMRItems(results)
		if len(items) != 3 {
			t.Fatalf("len(items) = %d, want 3", len(items))
		}
		if items[0].ID != "r1" || items[0].Topic != "f1" {
			t.Fatalf("item 0 = %#v", items[0])
		}
	})

	t.Run("Diversify selects diverse results", func(t *testing.T) {
		div := Diversify(results, 2, 0.5)
		if len(div) != 2 {
			t.Fatalf("len(div) = %d, want 2", len(div))
		}
		// r1 should be chosen first, and r3 should be preferred over r2 due to diversity (different direction)
		if div[0].ID != "r1" {
			t.Fatalf("first = %s, want r1", div[0].ID)
		}
		if div[1].ID != "r3" {
			t.Fatalf("second = %s, want r3", div[1].ID)
		}
	})

	t.Run("MMR empty edge cases", func(t *testing.T) {
		if got := MMR(nil, 5, 0.5); got != nil {
			t.Fatalf("MMR(nil) = %v, want nil", got)
		}
		items := MMRItems([]*Result{r1})
		if got := MMR(items, 0, 0.5); got != nil {
			t.Fatalf("MMR(k=0) = %v, want nil", got)
		}
	})

	t.Run("maxSimilarity handles missing embeddings", func(t *testing.T) {
		itemNoEmbed := MMRItem{ID: "no_embed", Score: 0.5}
		itemWithEmbed := MMRItem{ID: "embed", Score: 0.8, Embedding: []float64{1, 0}}

		sim1 := maxSimilarity(itemNoEmbed, []MMRItem{itemWithEmbed})
		if sim1 != 0 {
			t.Fatalf("sim1 = %v, want 0", sim1)
		}

		sim2 := maxSimilarity(itemWithEmbed, []MMRItem{itemNoEmbed})
		if sim2 != 0 {
			t.Fatalf("sim2 = %v, want 0", sim2)
		}
	})
}

func TestDiversifyWithTrace(t *testing.T) {
	t.Parallel()

	results := []*Result{
		{ID: "first", CombinedScore: 0.9, Embedding: []float64{1, 0}},
		{ID: "similar", CombinedScore: 0.85, Embedding: []float64{1, 0}},
		{ID: "diverse", CombinedScore: 0.7, Embedding: []float64{0, 1}},
	}

	got, traces := DiversifyWithTrace(results, 2, 0.9)
	want := Diversify(results, 2, 0.9)
	if len(got) != len(want) || len(traces) != len(got) {
		t.Fatalf("lengths: traced=%d untraced=%d traces=%d", len(got), len(want), len(traces))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("result %d = %q, want %q", i, got[i].ID, want[i].ID)
		}
	}
	if traces[0].Penalty != 0 || traces[0].MaxSimilarity != 0 {
		t.Fatalf("first trace = %#v, want zero similarity and penalty", traces[0])
	}
	if got[1].ID != "similar" {
		t.Fatalf("second result = %q, want similar", got[1].ID)
	}
	if traces[1].Penalty >= 0 {
		t.Fatalf("second penalty = %v, want negative", traces[1].Penalty)
	}
	wantPenalty := -(1 - traces[1].Lambda) * traces[1].MaxSimilarity
	if !closeFloat(traces[1].Penalty, wantPenalty) {
		t.Fatalf("second penalty = %v, want %v", traces[1].Penalty, wantPenalty)
	}
	wantSelection := traces[1].RelevanceContribution + traces[1].Penalty
	if !closeFloat(traces[1].SelectionScore, wantSelection) {
		t.Fatalf("second selection score = %v, want %v", traces[1].SelectionScore, wantSelection)
	}
}

func TestDiversifyWithTraceEdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("missing embeddings have zero similarity", func(t *testing.T) {
		results := []*Result{
			{ID: "embedded", CombinedScore: 0.9, Embedding: []float64{1, 0}},
			{ID: "missing", CombinedScore: 0.8},
		}
		got, traces := DiversifyWithTrace(results, 2, 0.5)
		if len(got) != 2 || got[1].ID != "missing" {
			t.Fatalf("results = %#v, want embedded then missing", got)
		}
		if traces[1].MaxSimilarity != 0 || traces[1].Penalty != 0 {
			t.Fatalf("missing-embedding trace = %#v, want zero similarity and penalty", traces[1])
		}
	})

	t.Run("negative cosine has zero similarity", func(t *testing.T) {
		results := []*Result{
			{ID: "first", CombinedScore: 0.9, Embedding: []float64{1, 0}},
			{ID: "opposite", CombinedScore: 0.8, Embedding: []float64{-1, 0}},
		}
		_, traces := DiversifyWithTrace(results, 2, 0.5)
		if traces[1].MaxSimilarity != 0 || traces[1].Penalty != 0 {
			t.Fatalf("opposite-embedding trace = %#v, want zero similarity and penalty", traces[1])
		}
	})

	t.Run("lambda is clamped", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			lambda float64
			want   float64
		}{
			{name: "negative", lambda: -1, want: 0},
			{name: "NaN", lambda: math.NaN(), want: 0},
			{name: "above one", lambda: 2, want: 1},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, traces := DiversifyWithTrace([]*Result{{ID: "one", CombinedScore: 0.5}}, 1, tc.lambda)
				if len(traces) != 1 || traces[0].Lambda != tc.want {
					t.Fatalf("traces = %#v, want lambda %v", traces, tc.want)
				}
			})
		}
	})

	t.Run("ties retain input order", func(t *testing.T) {
		results := []*Result{
			{ID: "a", CombinedScore: 0.8},
			{ID: "b", CombinedScore: 0.8},
			{ID: "c", CombinedScore: 0.8},
		}
		got, _ := DiversifyWithTrace(results, len(results), 0.5)
		for i := range results {
			if got[i] != results[i] {
				t.Fatalf("result %d = %q, want stable input %q", i, got[i].ID, results[i].ID)
			}
		}
	})

	t.Run("k limits trace and result count", func(t *testing.T) {
		results := []*Result{{ID: "a"}, nil, {ID: "b"}}
		got, traces := DiversifyWithTrace(results, 1, 0.5)
		if len(got) != 1 || len(traces) != 1 || got[0].ID != "a" {
			t.Fatalf("k=1 returned results=%#v traces=%#v", got, traces)
		}
		got, traces = DiversifyWithTrace(results, 5, 0.5)
		if len(got) != 2 || len(traces) != 2 {
			t.Fatalf("k=5 returned %d results and %d traces, want 2 each", len(got), len(traces))
		}
		got, traces = DiversifyWithTrace(results, 0, 0.5)
		if got != nil || traces != nil {
			t.Fatalf("k=0 returned results=%#v traces=%#v, want nil", got, traces)
		}
	})
}

func TestMMRPreservesNegativeScoreOrdering(t *testing.T) {
	t.Parallel()

	items := []MMRItem{
		{ID: "first", Score: 0},
		{ID: "minus-four", Score: -4},
		{ID: "minus-three", Score: -3},
	}
	got := MMR(items, len(items), 1)
	want := []string{"first", "minus-three", "minus-four"}
	for i, id := range want {
		if got[i].ID != id {
			t.Fatalf("MMR result %d = %q, want %q", i, got[i].ID, id)
		}
	}

	got = MMR(items[1:], 2, 1)
	if got[0].ID != "minus-three" || got[1].ID != "minus-four" {
		t.Fatalf("MMR first negative round = %#v, want -3 then -4", got)
	}
}

func TestMMRNegativeTiesRetainInputOrder(t *testing.T) {
	t.Parallel()

	items := []MMRItem{{ID: "a", Score: -3}, {ID: "b", Score: -3}, {ID: "c", Score: -3}}
	got := MMR(items, len(items), 1)
	for i := range items {
		if got[i].ID != items[i].ID {
			t.Fatalf("MMR result %d = %q, want stable input %q", i, got[i].ID, items[i].ID)
		}
	}
}

func TestMMROrdersNumericScoresAboveNaN(t *testing.T) {
	t.Parallel()

	items := []MMRItem{
		{ID: "nan", Score: math.NaN()},
		{ID: "negative-infinity", Score: math.Inf(-1)},
		{ID: "finite", Score: -4},
		{ID: "positive-infinity", Score: math.Inf(1)},
	}
	got := MMR(items, len(items), 1)
	want := []string{"positive-infinity", "finite", "negative-infinity", "nan"}
	for i, id := range want {
		if got[i].ID != id {
			t.Fatalf("MMR result %d = %q, want %q", i, got[i].ID, id)
		}
	}
}

func TestDiversifyAllNaNIsStableAndTraceAligned(t *testing.T) {
	t.Parallel()

	results := []*Result{
		{ID: "a", CombinedScore: math.NaN(), Embedding: []float64{1, 0}},
		{ID: "b", CombinedScore: math.NaN(), Embedding: []float64{1, 0}},
		{ID: "c", CombinedScore: math.NaN(), Embedding: []float64{0, 1}},
	}
	got, traces := DiversifyWithTrace(results, len(results), 0.5)
	want := Diversify(results, len(results), 0.5)
	if len(got) != len(results) || len(traces) != len(got) || len(want) != len(got) {
		t.Fatalf("lengths: traced=%d untraced=%d traces=%d want=%d", len(got), len(want), len(traces), len(results))
	}
	for i := range results {
		if got[i] != results[i] || want[i] != got[i] {
			t.Fatalf("result %d = traced %q untraced %q, want stable %q", i, got[i].ID, want[i].ID, results[i].ID)
		}
	}
	if traces[1].MaxSimilarity != 1 {
		t.Fatalf("second trace similarity = %v, want chosen item similarity 1", traces[1].MaxSimilarity)
	}
}

func closeFloat(got, want float64) bool {
	return math.Abs(got-want) <= 1e-12
}

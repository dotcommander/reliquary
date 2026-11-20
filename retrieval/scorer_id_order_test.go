package retrieval

import (
	"container/heap"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/dotcommander/reliquary/embedding"
)

func TestRerankTopByIDCalibratesFullPoolAndMatchesFullRanking(t *testing.T) {
	t.Parallel()

	makeResults := func() []*Result {
		return []*Result{
			{ID: "z", Content: "alpha", Embedding: []float64{1, 0}},
			{ID: "b", Content: "alpha", Embedding: []float64{1, 0}},
			{ID: "a", Content: "alpha", Embedding: []float64{1, 0}},
			{ID: "c", Content: "unrelated", Embedding: []float64{0, 1}},
		}
	}
	scorer := NewScorer(DefaultWeights())
	full := scorer.RerankEmbeddingByID(embedding.Vector{1, 0}, "alpha", makeResults())
	for _, limit := range []int{-1, 0, 1, 2, 3, 4, 5} {
		got := scorer.RerankTopByID([]float64{1, 0}, "alpha", makeResults(), limit)
		wantCount := len(full)
		if limit > 0 && limit < wantCount {
			wantCount = limit
		}
		if len(got) != wantCount {
			t.Fatalf("limit %d returned %d results, want %d", limit, len(got), wantCount)
		}
		for i, item := range got {
			if item.ID != full[i].ID || item.CombinedScore != full[i].CombinedScore || item.EmbeddingScore != full[i].EmbeddingScore || item.KeywordScore != full[i].KeywordScore {
				t.Fatalf("limit %d rank %d = %+v, want %+v", limit, i, item, full[i])
			}
		}
	}
}

func TestBoundedScoreIDOrderIncludesNaNAndTies(t *testing.T) {
	t.Parallel()

	items := []*Result{
		{ID: "nan-b", CombinedScore: math.NaN()},
		{ID: "b", CombinedScore: 2},
		{ID: "negative", CombinedScore: math.Inf(-1)},
		{ID: "a", CombinedScore: 2},
		{ID: "nan-a", CombinedScore: math.NaN()},
		{ID: "positive", CombinedScore: math.Inf(1)},
	}
	full := slices.Clone(items)
	slices.SortFunc(full, compareScoreID)
	for _, limit := range []int{1, 2, 3, 5} {
		best := make(worstFirstResults, 0, limit)
		for _, item := range items {
			if len(best) < limit {
				heap.Push(&best, item)
			} else if compareScoreID(item, best[0]) < 0 {
				best[0] = item
				heap.Fix(&best, 0)
			}
		}
		slices.SortFunc(best, compareScoreID)
		for i := range best {
			if best[i] != full[i] {
				t.Fatalf("limit %d rank %d = %q, want %q", limit, i, best[i].ID, full[i].ID)
			}
		}
	}
}

func TestRerankEmbeddingByIDPreservesScoringAndLegacyTieOrder(t *testing.T) {
	t.Parallel()

	makeResults := func() []*Result {
		return []*Result{
			{ID: "b", Content: "alpha", Embedding: []float64{1, 0}},
			{ID: "c", Content: "bravo", Embedding: []float64{0, 1}},
			{ID: "a", Content: "alpha", Embedding: []float64{1, 0}},
		}
	}
	scorer := NewScorer(DefaultWeights())
	query := embedding.Vector{1, 0}
	legacy := scorer.RerankEmbedding(query, "alpha", makeResults())
	plain := scorer.Rerank(EmbeddingVector(query), "alpha", makeResults())
	byID := scorer.RerankEmbeddingByID(query, "alpha", makeResults())

	for _, ranked := range [][]*Result{legacy, plain} {
		if ranked[0].ID != "b" || ranked[1].ID != "a" || ranked[2].ID != "c" {
			t.Fatalf("score-only order = %q, %q, %q, want b, a, c", ranked[0].ID, ranked[1].ID, ranked[2].ID)
		}
	}
	if byID[0].ID != "a" || byID[1].ID != "b" || byID[2].ID != "c" {
		t.Fatalf("score/ID order = %q, %q, %q, want a, b, c", byID[0].ID, byID[1].ID, byID[2].ID)
	}
	for _, ranked := range [][]*Result{legacy, plain, byID} {
		if ranked[0].CombinedScore != ranked[1].CombinedScore || ranked[1].CombinedScore <= ranked[2].CombinedScore {
			t.Fatalf("unexpected calibrated ranking scores: %v, %v, %v", ranked[0].CombinedScore, ranked[1].CombinedScore, ranked[2].CombinedScore)
		}
		for _, item := range ranked {
			var reference *Result
			for _, prior := range legacy {
				if prior.ID == item.ID {
					reference = prior
					break
				}
			}
			if reference == nil || item.EmbeddingScore != reference.EmbeddingScore || item.KeywordScore != reference.KeywordScore || item.FilenameScore != reference.FilenameScore || item.CombinedScore != reference.CombinedScore {
				t.Fatalf("scoring changed for %q: got %+v, reference %+v", item.ID, item, reference)
			}
		}
	}
}

func TestRecalibrateRawPreservesStableTieOrder(t *testing.T) {
	t.Parallel()

	const tiedCount = 24
	candidates := make([]RawCandidate, 0, tiedCount+2)
	candidates = append(candidates, RawCandidate{
		Result:  &Result{ID: "low"},
		Raw:     ScoreSignals{Embedding: 0.1},
		Present: SignalPresence{Embedding: true},
	})
	wantTiedIDs := make([]string, 0, tiedCount)
	for i := tiedCount - 1; i >= 0; i-- {
		id := fmt.Sprintf("tied-%02d", i)
		wantTiedIDs = append(wantTiedIDs, id)
		candidates = append(candidates, RawCandidate{
			Result:  &Result{ID: id},
			Raw:     ScoreSignals{Embedding: 0.5},
			Present: SignalPresence{Embedding: true},
		})
	}
	candidates = append(candidates, RawCandidate{
		Result:  &Result{ID: "high"},
		Raw:     ScoreSignals{Embedding: 0.9},
		Present: SignalPresence{Embedding: true},
	})

	scorer := NewScorer(DefaultWeights())
	got, _ := scorer.RecalibrateRaw("alpha", candidates, false, true)
	if len(got) != tiedCount+2 {
		t.Fatalf("len(got) = %d, want %d", len(got), tiedCount+2)
	}
	if got[0].ID != "high" || got[len(got)-1].ID != "low" {
		t.Fatalf("extremes = (%q, %q), want (high, low)", got[0].ID, got[len(got)-1].ID)
	}
	for i, wantID := range wantTiedIDs {
		if got[i+1].ID != wantID {
			t.Fatalf("tied position %d = %q, want %q (unstable sort reordered ties)", i, got[i+1].ID, wantID)
		}
	}
}

func TestCalibrateScoresIgnoresNonFiniteSignals(t *testing.T) {
	t.Parallel()

	scorer := NewScorer(DefaultWeights())
	results := []*Result{
		{ID: "nan-doc", Content: "alpha", Embedding: []float64{math.NaN(), 0}},
		{ID: "inf-doc", Content: "alpha", Embedding: []float64{math.Inf(1), 0}},
		{ID: "low-doc", Content: "alpha", Embedding: []float64{0, 1}},
		{ID: "high-doc", Content: "alpha", Embedding: []float64{1, 0}},
	}
	ranked := scorer.RerankEmbeddingByID(embedding.Vector{1, 0}, "alpha", results)

	byID := make(map[string]*Result, len(ranked))
	for _, r := range ranked {
		byID[r.ID] = r
		if math.IsNaN(r.EmbeddingScore) || math.IsInf(r.EmbeddingScore, 0) || math.IsNaN(r.CombinedScore) || math.IsInf(r.CombinedScore, 0) {
			t.Fatalf("result %q has non-finite calibrated score: %+v", r.ID, r)
		}
	}
	if byID["high-doc"].EmbeddingScore != 1.0 {
		t.Fatalf("high-doc EmbeddingScore = %v, want 1.0", byID["high-doc"].EmbeddingScore)
	}
	if byID["low-doc"].EmbeddingScore != 0.0 {
		t.Fatalf("low-doc EmbeddingScore = %v, want 0.0", byID["low-doc"].EmbeddingScore)
	}
	if byID["nan-doc"].EmbeddingScore != 0.0 || byID["inf-doc"].EmbeddingScore != 0.0 {
		t.Fatalf("non-finite docs EmbeddingScore = (%v, %v), want (0, 0)", byID["nan-doc"].EmbeddingScore, byID["inf-doc"].EmbeddingScore)
	}
	if ranked[0].ID != "high-doc" {
		t.Fatalf("top ranked ID = %q, want high-doc", ranked[0].ID)
	}
}

package retrieval

import (
	"math"
	"testing"
)

func TestScorerRerankDescending(t *testing.T) {
	t.Parallel()

	scorer := NewScorer(DefaultWeights())
	queryEmb := []float64{1.0, 0.0}
	results := []*Result{
		{ID: "no_filename", Content: "machine learning", Filename: "", Embedding: []float64{0.9, 0.1}},
		{ID: "matching_filename", Content: "data science", Filename: "machine-learning-guide.md", Embedding: []float64{0.7, 0.3}},
		{ID: "mismatched_filename", Content: "machine learning", Filename: "cooking-recipes.txt", Embedding: []float64{0.9, 0.1}},
	}
	for _, r := range results {
		norm := 0.0
		for _, x := range r.Embedding {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		for i := range r.Embedding {
			r.Embedding[i] /= norm
		}
	}

	scorer.Rerank(queryEmb, "machine learning", results)

	for i := 1; i < len(results); i++ {
		if results[i].CombinedScore > results[i-1].CombinedScore {
			t.Fatalf("not descending: index %d > %d", i, i-1)
		}
	}
}

// Recency/Importance must contribute nothing when their weights are 0 (the
// default), and must contribute exactly weight*value when set. This guards the
// "existing callers are byte-for-byte unchanged" invariant.
func TestScorerRecencyImportanceOptional(t *testing.T) {
	t.Parallel()

	// Zero weights (default): new signals are ignored even when populated.
	base := NewScorer(DefaultWeights())
	r := &Result{RecencyScore: 1.0, ImportanceScore: 1.0}
	got := base.Score(nil, "", r)
	if got != 0.0 {
		t.Fatalf("with zero recency/importance weights, score = %v, want 0", got)
	}

	// Non-zero weights: contribution is exactly weight*value, uncalibrated.
	w := DefaultWeights()
	w.Recency = 0.2
	w.Importance = 0.3
	s := NewScorer(w)
	r2 := &Result{RecencyScore: 0.5, ImportanceScore: 1.0}
	want := 0.2*0.5 + 0.3*1.0
	if got := s.Score(nil, "", r2); math.Abs(got-want) > 1e-9 {
		t.Fatalf("recency/importance contribution = %v, want %v", got, want)
	}
}

func TestScorerRerankPreservesRecencyImportanceWeights(t *testing.T) {
	t.Parallel()

	w := DefaultWeights()
	w.Recency = 0.2
	w.Importance = 0.3
	scorer := NewScorer(w)

	results := []*Result{
		{ID: "ordinary", Content: "alpha bravo", Embedding: []float64{1, 0}},
		{ID: "salient", Content: "alpha bravo", Embedding: []float64{1, 0}, RecencyScore: 1, ImportanceScore: 1},
	}

	ranked := scorer.Rerank([]float64{1, 0}, "alpha bravo", results)
	if ranked[0].ID != "salient" {
		t.Fatalf("Rerank() top result = %q, want salient", ranked[0].ID)
	}
	if ranked[0].CombinedScore <= ranked[1].CombinedScore {
		t.Fatalf("salience weights did not affect Rerank: salient=%v ordinary=%v", ranked[0].CombinedScore, ranked[1].CombinedScore)
	}
}

func TestScorerRerankLeavesAbsentSignalsAtZero(t *testing.T) {
	t.Parallel()

	scorer := NewScorer(DefaultWeights())
	results := []*Result{
		{ID: "a"},
		{ID: "b"},
	}

	ranked := scorer.Rerank(nil, "", results)
	for _, r := range ranked {
		if r.EmbeddingScore != 0 || r.KeywordScore != 0 || r.FilenameScore != 0 || r.CombinedScore != 0 {
			t.Fatalf("absent signals produced confidence for %q: %+v", r.ID, *r)
		}
	}
}

func TestScorerRerankCalibratesOnlyPresentSignals(t *testing.T) {
	t.Parallel()

	scorer := NewScorerWithOptions(Weights{Embedding: 1}, false)
	results := []*Result{
		{ID: "opposite", Embedding: []float64{-1, 0}},
		{ID: "negative", Embedding: []float64{-1, 1}},
		{ID: "absent"},
	}

	ranked, traces := scorer.RerankWithTrace([]float64{1, 0}, "", results)
	byID := make(map[string]*Result, len(ranked))
	traceByID := make(map[string]ScoreTrace, len(traces))
	for i, result := range ranked {
		byID[result.ID] = result
		traceByID[traces[i].ID] = traces[i]
	}

	if got := byID["negative"].EmbeddingScore; got != 1 {
		t.Fatalf("present negative signal calibrated score = %v, want 1", got)
	}
	if got := byID["absent"].EmbeddingScore; got != 0 {
		t.Fatalf("absent embedding calibrated score = %v, want 0", got)
	}
	if traceByID["absent"].Present.Embedding {
		t.Fatal("absent embedding reported as present")
	}
	if got := traceByID["absent"].Contributions.Embedding; got != 0 {
		t.Fatalf("absent embedding contribution = %v, want 0", got)
	}
}

func TestScorerClearsStaleComputedSignals(t *testing.T) {
	t.Parallel()

	scorer := NewScorerWithOptions(Weights{Embedding: 1, Keyword: 1, Filename: 1}, false)
	result := &Result{
		ID:             "reused",
		Content:        "alpha",
		Filename:       "alpha.md",
		Embedding:      []float64{1, 0},
		EmbeddingScore: 99,
		KeywordScore:   99,
		FilenameScore:  99,
		CombinedScore:  99,
	}

	scorer.Score([]float64{1, 0}, "alpha", result)
	result.Content = ""
	result.Filename = ""
	result.Embedding = nil
	scorer.Rerank(nil, "", []*Result{result})

	if result.EmbeddingScore != 0 || result.KeywordScore != 0 || result.FilenameScore != 0 || result.CombinedScore != 0 {
		t.Fatalf("reused result retained stale scores: %+v", *result)
	}
}

func TestScorerRerankHonorsCustomTextWeights(t *testing.T) {
	t.Parallel()

	w := Weights{Filename: 1.0}
	scorer := NewScorer(w)
	results := []*Result{
		{ID: "content_match", Content: "alpha"},
		{ID: "filename_match", Filename: "alpha.md"},
	}

	ranked := scorer.Rerank(nil, "alpha", results)
	if ranked[0].ID != "filename_match" {
		t.Fatalf("Rerank() top result = %q, want filename_match", ranked[0].ID)
	}
	if ranked[0].CombinedScore <= ranked[1].CombinedScore {
		t.Fatalf("custom filename weight did not dominate: filename=%v content=%v", ranked[0].CombinedScore, ranked[1].CombinedScore)
	}
}

func TestScorerRerankWithTraceExplainsScoresInRankOrder(t *testing.T) {
	t.Parallel()

	w := DefaultWeights()
	w.Recency = 0.2
	w.Importance = 0.3
	scorer := NewScorer(w)
	results := []*Result{
		{
			ID:              "ordinary",
			Content:         "alpha",
			Filename:        "ordinary.md",
			Embedding:       []float64{0, 1},
			RecencyScore:    0.0,
			ImportanceScore: 0.0,
		},
		{
			ID:              "best",
			Content:         "alpha bravo",
			Filename:        "alpha-bravo.md",
			Embedding:       []float64{1, 0},
			RecencyScore:    0.5,
			ImportanceScore: 1.0,
		},
	}

	ranked, traces := scorer.RerankWithTrace([]float64{1, 0}, "alpha bravo", results)

	if len(traces) != len(ranked) {
		t.Fatalf("trace count = %d, want %d", len(traces), len(ranked))
	}
	if ranked[0].ID != "best" || traces[0].ID != ranked[0].ID {
		t.Fatalf("trace order does not match ranked results: ranked[0]=%q trace[0]=%q", ranked[0].ID, traces[0].ID)
	}

	trace := traces[0]
	if !trace.AdaptiveWeights {
		t.Fatalf("AdaptiveWeights = false, want true for DefaultWeights scorer")
	}
	if trace.QueryTokenCount != 2 {
		t.Fatalf("QueryTokenCount = %d, want 2", trace.QueryTokenCount)
	}
	if trace.Weights != (Weights{Embedding: 0.45, Keyword: 0.45, Filename: 0.10, Recency: 0.2, Importance: 0.3}) {
		t.Fatalf("Weights = %+v, want adaptive text weights plus salience", trace.Weights)
	}
	if !trace.Present.Embedding || !trace.Present.Keyword || !trace.Present.Filename || !trace.Present.Recency || !trace.Present.Importance {
		t.Fatalf("Present flags missing active signals: %+v", trace.Present)
	}
	if math.Abs(trace.Raw.Embedding-1.0) > 1e-9 {
		t.Fatalf("Raw.Embedding = %v, want 1", trace.Raw.Embedding)
	}
	if math.Abs(trace.Calibrated.Embedding-1.0) > 1e-9 {
		t.Fatalf("Calibrated.Embedding = %v, want 1", trace.Calibrated.Embedding)
	}

	contributionSum := trace.Contributions.Embedding + trace.Contributions.Keyword + trace.Contributions.Filename +
		trace.Contributions.Recency + trace.Contributions.Importance
	if math.Abs(contributionSum-trace.CombinedScore) > 1e-9 {
		t.Fatalf("contribution sum = %v, CombinedScore = %v", contributionSum, trace.CombinedScore)
	}
	if math.Abs(trace.CombinedScore-ranked[0].CombinedScore) > 1e-9 {
		t.Fatalf("trace CombinedScore = %v, ranked CombinedScore = %v", trace.CombinedScore, ranked[0].CombinedScore)
	}
}

func TestScorerRerankWithTraceReportsAbsentSignalsAndCustomWeights(t *testing.T) {
	t.Parallel()

	w := Weights{Filename: 1.0}
	scorer := NewScorer(w)
	results := []*Result{
		{ID: "empty"},
		{ID: "filename", Filename: "alpha.md"},
	}

	ranked, traces := scorer.RerankWithTrace(nil, "alpha", results)

	if ranked[0].ID != "filename" || traces[0].ID != "filename" {
		t.Fatalf("top ranked/trace = %q/%q, want filename", ranked[0].ID, traces[0].ID)
	}
	if traces[0].AdaptiveWeights {
		t.Fatalf("AdaptiveWeights = true, want false for custom text weights")
	}
	if traces[0].Weights != w {
		t.Fatalf("Weights = %+v, want %+v", traces[0].Weights, w)
	}
	if traces[0].Present.Embedding || traces[0].Present.Keyword || !traces[0].Present.Filename {
		t.Fatalf("filename trace presence = %+v, want only filename text signal", traces[0].Present)
	}
	if traces[1].Present.Embedding || traces[1].Present.Keyword || traces[1].Present.Filename {
		t.Fatalf("empty trace presence = %+v, want no text/vector signals", traces[1].Present)
	}
}

func TestRecencyFromAge(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		age, halfLife float64
		want          float64
	}{
		{"now is fresh", 0, 100, 1.0},
		{"future is fresh", -10, 100, 1.0},
		{"no halflife means no decay", 50, 0, 1.0},
		{"one halflife is half", 100, 100, 0.5},
		{"two halflives is quarter", 200, 100, 0.25},
		{"nan age has zero freshness", math.NaN(), 100, 0},
		{"nan age overrides disabled decay", math.NaN(), 0, 0},
		{"nan halflife has zero freshness", 100, math.NaN(), 0},
		{"nan halflife overrides fresh age", 0, math.NaN(), 0},
		{"both nan have zero freshness", math.NaN(), math.NaN(), 0},
		{"infinite ratio has zero freshness", math.Inf(1), math.Inf(1), 0},
		{"infinite age has zero freshness", math.Inf(1), 100, 0},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := RecencyFromAge(tc.age, tc.halfLife); math.Abs(got-tc.want) > 1e-9 {
				t.Fatalf("RecencyFromAge(%v,%v) = %v, want %v", tc.age, tc.halfLife, got, tc.want)
			}
		})
	}
}

func TestRecencyFromAgeAlwaysReturnsFiniteUnitRange(t *testing.T) {
	t.Parallel()

	for _, age := range []float64{math.Inf(-1), -1, 0, 1, math.MaxFloat64, math.Inf(1), math.NaN()} {
		for _, halfLife := range []float64{math.Inf(-1), -1, 0, 1, math.MaxFloat64, math.Inf(1), math.NaN()} {
			got := RecencyFromAge(age, halfLife)
			if math.IsNaN(got) || math.IsInf(got, 0) || got < 0 || got > 1 {
				t.Fatalf("RecencyFromAge(%v, %v) = %v, want finite value in [0,1]", age, halfLife, got)
			}
		}
	}
}

func TestScorerDoesNotPropagateNaNRecency(t *testing.T) {
	t.Parallel()

	result := &Result{ID: "nan-recency", RecencyScore: RecencyFromAge(math.Inf(1), math.Inf(1))}
	NewScorerWithOptions(Weights{Recency: 1}, false).Rerank(nil, "", []*Result{result})
	if math.IsNaN(result.CombinedScore) || result.CombinedScore != 0 {
		t.Fatalf("CombinedScore = %v, want 0", result.CombinedScore)
	}
}

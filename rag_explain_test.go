package reliquary_test

import (
	"context"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestWithExplainIsOptInAndDoesNotChangeSearch(t *testing.T) {
	stored := rerankerCandidates()
	stored[0].Explain = &retrieval.SearchExplanation{FinalRank: 99}
	app, _ := newRerankerTestApp(t, stored)

	plain, err := app.Search(context.Background(), "query match")
	if err != nil {
		t.Fatalf("plain Search: %v", err)
	}
	explained, err := app.Search(context.Background(), "query match", reliquary.WithExplain())
	if err != nil {
		t.Fatalf("explained Search: %v", err)
	}
	if len(plain) != len(explained) {
		t.Fatalf("result counts = %d and %d", len(plain), len(explained))
	}
	for i := range plain {
		if plain[i].Explain != nil {
			t.Fatalf("plain result %d explanation = %#v, want nil", i, plain[i].Explain)
		}
		if plain[i].ID != explained[i].ID || plain[i].CombinedScore != explained[i].CombinedScore {
			t.Fatalf("explanation changed result %d: plain=%#v explained=%#v", i, plain[i], explained[i])
		}
		if explained[i].Explain == nil || explained[i].Explain.FinalRank != i+1 {
			t.Fatalf("explained result %d = %#v", i, explained[i].Explain)
		}
		if explained[i].Explain.HybridRank != i+1 || !explained[i].Explain.HybridScoreUsed {
			t.Fatalf("hybrid explanation %d = %#v", i, explained[i].Explain)
		}
	}
	if stored[0].Explain.FinalRank != 99 {
		t.Fatalf("stored explanation mutated: %#v", stored[0].Explain)
	}
}

func TestWithExplainHybridTraceUsesRawAndCalibratedSignals(t *testing.T) {
	app, _ := newRerankerTestApp(t, []*retrieval.Result{
		{ID: "strong", Content: "alpha beta", Filename: "alpha.md", Embedding: []float64{1, 0}},
		{ID: "weak", Content: "other", Filename: "other.md", Embedding: []float64{-1, 0}},
	})
	hits, err := app.Search(context.Background(), "alpha beta", reliquary.WithExplain())
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, hit := range hits {
		trace := hit.Explain.Hybrid
		sum := trace.Contributions.Embedding + trace.Contributions.Keyword + trace.Contributions.Filename + trace.Contributions.Recency + trace.Contributions.Importance
		if math.Abs(sum-trace.CombinedScore) > 1e-12 {
			t.Fatalf("%s contribution sum %v != %v", hit.ID, sum, trace.CombinedScore)
		}
		if trace.Weights != retrieval.AdaptiveWeights(2) || !trace.AdaptiveWeights {
			t.Fatalf("%s effective weights = %#v", hit.ID, trace)
		}
	}
	if got := hits[0].Explain.Hybrid.Raw.Keyword; got != 1 {
		t.Fatalf("raw keyword overlap = %v, want 1", got)
	}
	if got := hits[1].Explain.Hybrid.Raw.Embedding; got != -1 {
		t.Fatalf("raw vector similarity = %v, want -1", got)
	}
	if got := hits[1].Explain.Hybrid.Calibrated.Embedding; got != 0 {
		t.Fatalf("calibrated vector similarity = %v, want 0", got)
	}
}

func TestWithExplainHonorsCustomWeights(t *testing.T) {
	custom := retrieval.Weights{Filename: 1}
	index := &recordingIndex{results: []*retrieval.Result{
		{ID: "content", Content: "alpha", Filename: "other.md", Embedding: []float64{1, 0}},
		{ID: "filename", Content: "other", Filename: "alpha.md", Embedding: []float64{0, 1}},
	}}
	app, err := reliquary.New(
		reliquary.WithEmbedder(rerankerTestEmbedder{}),
		reliquary.WithIndex(index),
		reliquary.WithIndexIdentity("custom-weight-test"),
		reliquary.WithWeights(custom),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	hits, err := app.Search(context.Background(), "alpha", reliquary.WithExplain())
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if hits[0].ID != "filename" {
		t.Fatalf("top result = %q, want filename", hits[0].ID)
	}
	for _, hit := range hits {
		trace := hit.Explain.Hybrid
		if trace.Weights != custom || trace.AdaptiveWeights {
			t.Fatalf("%s trace weights = %#v", hit.ID, trace)
		}
	}
}

func TestWithExplainRRFAndRerankerStageOwnership(t *testing.T) {
	idx := &rrfIndex{rows: [][]*retrieval.Result{
		{{ID: "a", Content: "query", Embedding: []float64{1, 0}}, {ID: "b", Embedding: []float64{0, 1}}},
		{{ID: "b", Content: "query"}, {ID: "a", Content: "query"}},
	}}
	reranker := &scriptedReranker{scores: []float64{0.2, 0.9}}
	hits, err := newRRFApp(t, idx).Search(context.Background(), "query",
		reliquary.WithRRF(60), reliquary.WithReranker(reranker), reliquary.WithExplain())
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got := resultIDs(hits); !reflect.DeepEqual(got, []string{"b", "a"}) {
		t.Fatalf("IDs = %v, want [b a]", got)
	}
	byID := map[string]*retrieval.Result{hits[0].ID: hits[0], hits[1].ID: hits[1]}
	a := byID["a"].Explain
	b := byID["b"].Explain
	if a.HybridScoreUsed || b.HybridScoreUsed {
		t.Fatal("hybrid score marked authoritative on RRF path")
	}
	if a.RRF.VectorRank != 1 || a.RRF.LexicalRank != 2 || a.RRF.FusedRank != 1 {
		t.Fatalf("a RRF = %#v", a.RRF)
	}
	if b.RRF.VectorRank != 2 || b.RRF.LexicalRank != 1 || b.RRF.FusedRank != 2 {
		t.Fatalf("b RRF = %#v", b.RRF)
	}
	for _, explanation := range []*retrieval.SearchExplanation{a, b} {
		rrf := explanation.RRF
		if math.Abs(rrf.VectorContribution+rrf.LexicalContribution-rrf.FusedScore) > 1e-12 {
			t.Fatalf("RRF contributions = %#v", rrf)
		}
	}
	if a.Reranker.InputRank != 1 || a.Reranker.Score != 0.2 || a.Reranker.Rank != 2 || a.FinalRank != 2 {
		t.Fatalf("a reranker/final = %#v", a)
	}
	if b.Reranker.InputRank != 2 || b.Reranker.Score != 0.9 || b.Reranker.Rank != 1 || b.FinalRank != 1 {
		t.Fatalf("b reranker/final = %#v", b)
	}
}

func TestWithExplainRRFAbsentLaneHasZeroRankAndContribution(t *testing.T) {
	idx := &rrfIndex{rows: [][]*retrieval.Result{
		{{ID: "a", Embedding: []float64{1, 0}}},
		{{ID: "b", Content: "query"}},
	}}
	hits, err := newRRFApp(t, idx).Search(context.Background(), "query", reliquary.WithRRF(60), reliquary.WithExplain())
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got := resultIDs(hits); !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("IDs = %v, want [a b]", got)
	}
	a := hits[0].Explain.RRF
	if a.VectorRank != 1 || a.LexicalRank != 0 || a.VectorContribution != 0.5 || a.LexicalContribution != 0 || a.FusedScore != 0.5 {
		t.Fatalf("a RRF = %#v", a)
	}
	b := hits[1].Explain.RRF
	if b.VectorRank != 0 || b.LexicalRank != 1 || b.VectorContribution != 0 || b.LexicalContribution != 0.5 || b.FusedScore != 0.5 {
		t.Fatalf("b RRF = %#v", b)
	}
}

func TestWithExplainMMRAndTopK(t *testing.T) {
	app, _ := newRerankerTestApp(t, rerankerCandidates())
	hits, err := app.Search(context.Background(), "query match", reliquary.WithMMR(0.5), reliquary.TopK(2), reliquary.WithExplain())
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("len(hits) = %d, want 2", len(hits))
	}
	for i, hit := range hits {
		if hit.Explain.MMR == nil || hit.Explain.FinalRank != i+1 {
			t.Fatalf("hit %d explanation = %#v", i, hit.Explain)
		}
		if hit.CombinedScore != hit.Explain.MMR.Relevance {
			t.Fatalf("MMR changed relevance for %s: %v vs %#v", hit.ID, hit.CombinedScore, hit.Explain.MMR)
		}
	}
	if hits[0].Explain.MMR.Penalty != 0 {
		t.Fatalf("first MMR penalty = %v, want 0", hits[0].Explain.MMR.Penalty)
	}
}

func TestSearchBatchWithExplainBlanksDuplicatesAndReusableOption(t *testing.T) {
	ctx := context.Background()
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, document.Document{ID: "doc", Text: "alpha beta gamma"}); err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	option := reliquary.WithExplain()
	rows, err := app.SearchBatch(ctx, []string{"alpha", " ", "alpha"}, option)
	if err != nil {
		t.Fatalf("SearchBatch: %v", err)
	}
	if rows[1] != nil || len(rows[0]) == 0 || len(rows[2]) == 0 {
		t.Fatalf("rows = %#v", rows)
	}
	if rows[0][0].Explain == nil || rows[2][0].Explain == nil || rows[0][0].Explain == rows[2][0].Explain {
		t.Fatalf("duplicate query explanations alias or are nil: %#v %#v", rows[0][0].Explain, rows[2][0].Explain)
	}
	individual, err := app.Search(ctx, "alpha", option)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if !reflect.DeepEqual(rows[0], individual) {
		t.Fatalf("batch explanation = %#v, individual = %#v", rows[0], individual)
	}

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			hits, err := app.Search(ctx, "alpha", option)
			if err != nil || len(hits) == 0 || hits[0].Explain == nil {
				t.Errorf("concurrent Search = %#v, %v", hits, err)
			}
		}()
	}
	wg.Wait()
}

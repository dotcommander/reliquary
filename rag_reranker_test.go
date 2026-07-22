package reliquary_test

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

type rerankCall struct {
	query      string
	candidates []*retrieval.Result
	contextVal any
}

type scriptedReranker struct {
	calls     []rerankCall
	scores    []float64
	scoreRows [][]float64
	err       error
	failAt    int
	cancel    context.CancelFunc
	cancelAt  int
	mutate    bool
	active    int
	maxActive int
}

func (r *scriptedReranker) Rerank(ctx context.Context, query string, candidates []*retrieval.Result) ([]float64, error) {
	r.active++
	if r.active > r.maxActive {
		r.maxActive = r.active
	}
	defer func() { r.active-- }()

	snapshot := make([]*retrieval.Result, len(candidates))
	for i, candidate := range candidates {
		snapshot[i] = cloneRerankCandidate(candidate)
	}
	r.calls = append(r.calls, rerankCall{query: query, candidates: snapshot, contextVal: ctx.Value(rerankerContextKey{})})
	call := len(r.calls)
	if r.mutate && len(candidates) > 0 {
		candidates[0].ID = "mutated"
		candidates[0].Content = "mutated"
		candidates[0].CombinedScore = -1
		if len(candidates[0].Embedding) > 0 {
			candidates[0].Embedding[0] = 99
		}
		if candidates[0].Metadata != nil {
			candidates[0].Metadata["key"] = "mutated"
			candidates[0].Metadata["nested"].(map[string]any)["value"] = "mutated"
			candidates[0].Metadata["array"].([]any)[0].(map[string]any)["value"] = "mutated"
		}
	}
	if r.cancelAt == call {
		r.cancel()
	}
	if r.failAt == call {
		return nil, r.err
	}
	if call <= len(r.scoreRows) {
		return slices.Clone(r.scoreRows[call-1]), nil
	}
	return slices.Clone(r.scores), nil
}

func cloneRerankCandidate(candidate *retrieval.Result) *retrieval.Result {
	clone := *candidate
	clone.Embedding = slices.Clone(candidate.Embedding)
	if candidate.Metadata != nil {
		clone.Metadata = make(map[string]any, len(candidate.Metadata))
		for key, value := range candidate.Metadata {
			clone.Metadata[key] = value
		}
	}
	return &clone
}

type rerankerContextKey struct{}

type rerankerTestEmbedder struct{}

func (rerankerTestEmbedder) Embed(_ context.Context, request embedding.Request) (embedding.Result, error) {
	vectors := make([]embedding.Vector, len(request.Inputs))
	for i := range vectors {
		vectors[i] = embedding.Vector{1, 0}
	}
	return embedding.Result{Vectors: vectors}, nil
}

func newRerankerTestApp(t *testing.T, results []*retrieval.Result) (*reliquary.App, *recordingIndex) {
	t.Helper()
	index := &recordingIndex{results: results}
	app, err := reliquary.New(
		reliquary.WithEmbedder(rerankerTestEmbedder{}),
		reliquary.WithIndex(index),
		reliquary.WithIndexIdentity("reranker-test"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return app, index
}

func rerankerCandidates() []*retrieval.Result {
	return []*retrieval.Result{
		{ID: "hybrid-first", Content: "query match", Filename: "query.md", Metadata: map[string]any{
			"key":    "value",
			"nested": map[string]any{"value": "original"},
			"array":  []any{map[string]any{"value": "original"}},
		}, Embedding: []float64{1, 0}},
		{ID: "hybrid-second", Content: "unrelated", Filename: "other.md", Embedding: []float64{0, 1}},
		{ID: "hybrid-third", Content: "distant", Filename: "third.md", Embedding: []float64{-1, 0}},
	}
}

func TestSearchWithoutRerankerParity(t *testing.T) {
	app, _ := newRerankerTestApp(t, rerankerCandidates())
	want, err := app.Search(context.Background(), "query match")
	if err != nil {
		t.Fatalf("Search without reranker: %v", err)
	}

	var typedNil *scriptedReranker
	for _, option := range []reliquary.SearchOption{
		reliquary.WithReranker(nil),
		reliquary.WithReranker(typedNil),
	} {
		got, err := app.Search(context.Background(), "query match", option)
		if err != nil {
			t.Fatalf("Search with disabled reranker: %v", err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("disabled reranker results = %#v, want %#v", got, want)
		}
	}
}

func TestWithRerankerLastOccurrenceWins(t *testing.T) {
	app, _ := newRerankerTestApp(t, rerankerCandidates()[:1])
	failing := &scriptedReranker{err: errors.New("must be disabled"), failAt: 1}
	if _, err := app.Search(context.Background(), "query", reliquary.WithReranker(failing), reliquary.WithReranker(nil)); err != nil {
		t.Fatalf("Search with final nil reranker: %v", err)
	}
	if len(failing.calls) != 0 {
		t.Fatalf("disabled reranker calls = %d, want 0", len(failing.calls))
	}

	enabled := &scriptedReranker{scores: []float64{0.75}}
	if _, err := app.Search(context.Background(), "query", reliquary.WithReranker(nil), reliquary.WithReranker(enabled)); err != nil {
		t.Fatalf("Search with final enabled reranker: %v", err)
	}
	if len(enabled.calls) != 1 {
		t.Fatalf("enabled reranker calls = %d, want 1", len(enabled.calls))
	}
}

func TestSearchRerankerReceivesHybridRankingAndOwnsFinalScore(t *testing.T) {
	app, index := newRerankerTestApp(t, rerankerCandidates())
	reranker := &scriptedReranker{scores: []float64{0.2, 0.9, 0.1}}
	hits, err := app.Search(
		context.Background(),
		"query match",
		reliquary.CandidateLimit(50),
		reliquary.WithReranker(reranker),
		reliquary.TopK(2),
	)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if index.lastQuery.Limit != 50 {
		t.Fatalf("candidate limit = %d, want 50", index.lastQuery.Limit)
	}
	if len(reranker.calls) != 1 {
		t.Fatalf("reranker calls = %d, want 1", len(reranker.calls))
	}
	candidates := reranker.calls[0].candidates
	if got := []string{candidates[0].ID, candidates[1].ID, candidates[2].ID}; !reflect.DeepEqual(got, []string{"hybrid-first", "hybrid-second", "hybrid-third"}) {
		t.Fatalf("reranker input order = %v, want hybrid order", got)
	}
	if candidates[0].EmbeddingScore <= candidates[1].EmbeddingScore || candidates[0].KeywordScore <= candidates[1].KeywordScore || candidates[0].CombinedScore <= candidates[1].CombinedScore {
		t.Fatalf("reranker inputs do not contain hybrid component scores: %#v", candidates)
	}
	if len(hits) != 2 || hits[0].ID != "hybrid-second" || hits[1].ID != "hybrid-first" {
		t.Fatalf("reranked hits = %#v, want [hybrid-second hybrid-first]", hits)
	}
	if hits[0].CombinedScore != 0.9 || hits[1].CombinedScore != 0.2 {
		t.Fatalf("final scores = [%v %v], want [0.9 0.2]", hits[0].CombinedScore, hits[1].CombinedScore)
	}
	if hits[0].EmbeddingScore != candidates[1].EmbeddingScore || hits[0].KeywordScore != candidates[1].KeywordScore {
		t.Fatalf("external rerank changed hybrid components: hit=%#v input=%#v", hits[0], candidates[1])
	}
}

func TestSearchRerankerStableTiesPreserveHybridOrder(t *testing.T) {
	app, _ := newRerankerTestApp(t, rerankerCandidates())
	reranker := &scriptedReranker{scores: []float64{0.5, 0.5, 0.5}}
	hits, err := app.Search(context.Background(), "query match", reliquary.WithReranker(reranker))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	got := []string{hits[0].ID, hits[1].ID, hits[2].ID}
	want := []string{"hybrid-first", "hybrid-second", "hybrid-third"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stable tie order = %v, want %v", got, want)
	}
}

func TestSearchMMRUsesRerankerScoreAsRelevance(t *testing.T) {
	app, _ := newRerankerTestApp(t, rerankerCandidates())
	reranker := &scriptedReranker{scores: []float64{0.1, 1, 0.2}}
	hits, err := app.Search(context.Background(), "query match", reliquary.WithReranker(reranker), reliquary.TopK(2), reliquary.WithMMR(1))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 || hits[0].ID != "hybrid-second" || hits[0].CombinedScore != 1 {
		t.Fatalf("MMR results = %#v, want reranker top result first", hits)
	}
}

func TestSearchRerankerContextAndMutationIsolation(t *testing.T) {
	stored := rerankerCandidates()[0]
	app, _ := newRerankerTestApp(t, []*retrieval.Result{stored})
	reranker := &scriptedReranker{scores: []float64{0.8}, mutate: true}
	ctx := context.WithValue(context.Background(), rerankerContextKey{}, "request-value")
	hits, err := app.Search(ctx, "query", reliquary.WithReranker(reranker))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if reranker.calls[0].contextVal != "request-value" {
		t.Fatalf("reranker context value = %#v, want request-value", reranker.calls[0].contextVal)
	}
	if hits[0].ID != "hybrid-first" || hits[0].Content != "query match" || hits[0].Embedding[0] != 1 || hits[0].Metadata["key"] != "value" || hits[0].CombinedScore != 0.8 {
		t.Fatalf("returned hit was mutated by reranker: %#v", hits[0])
	}
	if hits[0].Metadata["nested"].(map[string]any)["value"] != "original" || hits[0].Metadata["array"].([]any)[0].(map[string]any)["value"] != "original" {
		t.Fatalf("returned hit nested metadata was mutated by reranker: %#v", hits[0].Metadata)
	}
	if stored.ID != "hybrid-first" || stored.Content != "query match" || stored.Embedding[0] != 1 || stored.Metadata["key"] != "value" {
		t.Fatalf("stored candidate was mutated by reranker: %#v", stored)
	}
	if stored.Metadata["nested"].(map[string]any)["value"] != "original" || stored.Metadata["array"].([]any)[0].(map[string]any)["value"] != "original" {
		t.Fatalf("stored candidate nested metadata was mutated by reranker: %#v", stored.Metadata)
	}
}

func TestSearchRerankerFailureReturnsNoResults(t *testing.T) {
	want := errors.New("reranker failed")
	tests := []struct {
		name     string
		reranker *scriptedReranker
		wantErr  error
	}{
		{name: "provider error", reranker: &scriptedReranker{err: want, failAt: 1}, wantErr: want},
		{name: "malformed scores", reranker: &scriptedReranker{scores: []float64{0.5}}, wantErr: retrieval.ErrInvalidRerankResult},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			app, _ := newRerankerTestApp(t, rerankerCandidates())
			hits, err := app.Search(context.Background(), "query", reliquary.WithReranker(tt.reranker))
			if !errors.Is(err, tt.wantErr) || hits != nil {
				t.Fatalf("Search = %#v, %v, want nil, %v", hits, err, tt.wantErr)
			}
		})
	}
}

func TestSearchRerankerCancellationIsFatal(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	reranker := &scriptedReranker{scores: []float64{0.8}, cancel: cancel, cancelAt: 1}
	app, _ := newRerankerTestApp(t, rerankerCandidates()[:1])
	hits, err := app.Search(ctx, "query", reliquary.WithReranker(reranker))
	if !errors.Is(err, context.Canceled) || hits != nil {
		t.Fatalf("Search = %#v, %v, want nil, context.Canceled", hits, err)
	}
}

func TestSearchSkipsRerankerForBlankQueryAndEmptyCandidates(t *testing.T) {
	reranker := &scriptedReranker{err: errors.New("must not run"), failAt: 1}
	app, _ := newRerankerTestApp(t, nil)
	for _, query := range []string{" ", "query"} {
		if hits, err := app.Search(context.Background(), query, reliquary.WithReranker(reranker)); err != nil || hits != nil {
			t.Fatalf("Search(%q) = %#v, %v, want nil, nil", query, hits, err)
		}
	}
	if len(reranker.calls) != 0 {
		t.Fatalf("reranker calls = %d, want 0", len(reranker.calls))
	}
}

func TestSearchBatchReranksSequentiallyAndPreservesRows(t *testing.T) {
	app, _ := newRerankerTestApp(t, rerankerCandidates()[:1])
	reranker := &scriptedReranker{scoreRows: [][]float64{{0.1}, {0.2}, {0.3}}}
	rows, err := app.SearchBatch(context.Background(), []string{"first", " ", "first", "third"}, reliquary.WithReranker(reranker))
	if err != nil {
		t.Fatalf("SearchBatch: %v", err)
	}
	if len(rows) != 4 || rows[1] != nil || rows[0][0].CombinedScore != 0.1 || rows[2][0].CombinedScore != 0.2 || rows[3][0].CombinedScore != 0.3 {
		t.Fatalf("batch rows = %#v, want aligned reranked rows", rows)
	}
	queries := []string{reranker.calls[0].query, reranker.calls[1].query, reranker.calls[2].query}
	if !reflect.DeepEqual(queries, []string{"first", "first", "third"}) {
		t.Fatalf("reranker query calls = %v", queries)
	}
	if reranker.maxActive != 1 {
		t.Fatalf("maximum concurrent reranker calls = %d, want 1", reranker.maxActive)
	}
}

func TestSearchBatchRerankerStopsOnFirstFailure(t *testing.T) {
	want := errors.New("second rerank failed")
	reranker := &scriptedReranker{scoreRows: [][]float64{{0.1}}, err: want, failAt: 2}
	app, _ := newRerankerTestApp(t, rerankerCandidates()[:1])
	rows, err := app.SearchBatch(context.Background(), []string{"one", "two", "three"}, reliquary.WithReranker(reranker))
	if !errors.Is(err, want) || rows != nil {
		t.Fatalf("SearchBatch = %#v, %v, want nil, %v", rows, err, want)
	}
	if len(reranker.calls) != 2 {
		t.Fatalf("reranker calls = %d, want 2", len(reranker.calls))
	}
}

func TestSearchBatchRerankerCancellationStopsWithoutPartialRows(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	reranker := &scriptedReranker{scores: []float64{0.1}, cancel: cancel, cancelAt: 1}
	app, _ := newRerankerTestApp(t, rerankerCandidates()[:1])
	rows, err := app.SearchBatch(ctx, []string{"one", "two"}, reliquary.WithReranker(reranker))
	if !errors.Is(err, context.Canceled) || rows != nil {
		t.Fatalf("SearchBatch = %#v, %v, want nil, context.Canceled", rows, err)
	}
	if len(reranker.calls) != 1 {
		t.Fatalf("reranker calls = %d, want 1", len(reranker.calls))
	}
}

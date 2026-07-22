package reliquary_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

type batchRecordingEmbedder struct {
	calls    int
	requests []embedding.Request
	result   embedding.Result
	err      error
}

func (e *batchRecordingEmbedder) Embed(_ context.Context, request embedding.Request) (embedding.Result, error) {
	e.calls++
	e.requests = append(e.requests, request)
	if e.err != nil || e.result.Vectors != nil {
		return e.result, e.err
	}
	vectors := make([]embedding.Vector, len(request.Inputs))
	for i := range request.Inputs {
		vectors[i] = embedding.Vector{float32(i + 1), 1}
	}
	return embedding.Result{Vectors: vectors}, nil
}

type batchRecordingIndex struct {
	queries       []reliquary.IndexQuery
	filterValues  []any
	results       []*retrieval.Result
	failAt        int
	err           error
	mutateFilters bool
	mutateVectors bool
	cancelAt      int
	cancel        context.CancelFunc
	active        int
	maxActive     int
	vectorValues  []float32
}

func (*batchRecordingIndex) Upsert(context.Context, []*retrieval.Result) error { return nil }
func (*batchRecordingIndex) ReplaceDocuments(context.Context, []reliquary.DocumentReplacement) error {
	return nil
}
func (*batchRecordingIndex) DeleteDocument(context.Context, string) error { return nil }
func (i *batchRecordingIndex) Search(ctx context.Context, query reliquary.IndexQuery) ([]*retrieval.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	i.active++
	if i.active > i.maxActive {
		i.maxActive = i.active
	}
	defer func() { i.active-- }()
	i.queries = append(i.queries, query)
	if len(query.Vector) > 0 {
		i.vectorValues = append(i.vectorValues, query.Vector[0])
	}
	if query.Filter != nil {
		i.filterValues = append(i.filterValues, query.Filter["tenant"])
	}
	if i.mutateFilters && query.Filter != nil {
		query.Filter["tenant"] = "mutated"
	}
	if i.mutateVectors && len(query.Vector) > 0 {
		query.Vector[0] = 99
	}
	if i.cancelAt > 0 && len(i.queries) == i.cancelAt {
		i.cancel()
	}
	if i.failAt > 0 && len(i.queries) == i.failAt {
		return i.results, i.err
	}
	return i.results, nil
}

func newBatchTestApp(t *testing.T, embedder embedding.Embedder, index reliquary.Index) *reliquary.App {
	t.Helper()
	app, err := reliquary.New(
		reliquary.WithEmbedder(embedder),
		reliquary.WithIndex(index),
		reliquary.WithIndexIdentity("batch-test"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return app
}

func TestSearchBatchEmbedsNonblankQueriesOnceAndPreservesAlignment(t *testing.T) {
	embedder := &batchRecordingEmbedder{}
	index := &batchRecordingIndex{results: []*retrieval.Result{{ID: "hit", Content: "content", Embedding: []float64{1, 1}}}}
	app := newBatchTestApp(t, embedder, index)

	rows, err := app.SearchBatch(context.Background(), []string{"first", "  ", "first", "third"}, reliquary.TopK(1))
	if err != nil {
		t.Fatalf("SearchBatch: %v", err)
	}
	if embedder.calls != 1 || len(embedder.requests) != 1 {
		t.Fatalf("embed calls = %d, requests = %d, want one", embedder.calls, len(embedder.requests))
	}
	if got, want := embedder.requests[0].Inputs, []string{"first", "first", "third"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("embedded inputs = %#v, want %#v", got, want)
	}
	if len(rows) != 4 || rows[1] != nil || len(rows[0]) != 1 || len(rows[2]) != 1 || len(rows[3]) != 1 {
		t.Fatalf("rows = %#v, want aligned nonblank results", rows)
	}
	if len(index.queries) != 3 {
		t.Fatalf("index calls = %d, want 3", len(index.queries))
	}
	for i, want := range []string{"first", "first", "third"} {
		if index.queries[i].Text != want || index.queries[i].Vector[0] != float32(i+1) {
			t.Fatalf("query %d = %#v", i, index.queries[i])
		}
	}
	if index.maxActive != 1 {
		t.Fatalf("maximum concurrent index calls = %d, want 1", index.maxActive)
	}
}

func TestSearchBatchAllBlankDoesNoIO(t *testing.T) {
	embedder := &batchRecordingEmbedder{err: errors.New("must not embed")}
	index := &batchRecordingIndex{err: errors.New("must not search"), failAt: 1}
	app := newBatchTestApp(t, embedder, index)

	rows, err := app.SearchBatch(context.Background(), []string{"", " \t"}, reliquary.WithFilter(map[string]any{"bad": []string{"ignored"}}))
	if err != nil {
		t.Fatalf("SearchBatch all blank: %v", err)
	}
	if len(rows) != 2 || rows[0] != nil || rows[1] != nil {
		t.Fatalf("rows = %#v, want two nil rows", rows)
	}
	if embedder.calls != 0 || len(index.queries) != 0 {
		t.Fatalf("I/O calls = embed %d, index %d", embedder.calls, len(index.queries))
	}
}

func TestSearchBatchValidatesCompleteEmbeddingBeforeIndexAccess(t *testing.T) {
	tests := []struct {
		name   string
		result embedding.Result
	}{
		{
			name:   "wrong vector count",
			result: embedding.Result{Vectors: []embedding.Vector{{1, 0}}},
		},
		{
			name:   "invalid later vector",
			result: embedding.Result{Vectors: []embedding.Vector{{1, 0}, {float32(math.Inf(1)), 0}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedder := &batchRecordingEmbedder{result: tt.result}
			index := &batchRecordingIndex{}
			app := newBatchTestApp(t, embedder, index)

			rows, err := app.SearchBatch(context.Background(), []string{"valid", "invalid"})
			if !errors.Is(err, embedding.ErrInvalidResult) {
				t.Fatalf("SearchBatch error = %v, want ErrInvalidResult", err)
			}
			if rows != nil || len(index.queries) != 0 {
				t.Fatalf("rows/index calls = %#v/%d, want nil/0", rows, len(index.queries))
			}
		})
	}
}

func TestSearchBatchClonesFilterForEveryIndexCall(t *testing.T) {
	embedder := &batchRecordingEmbedder{}
	index := &batchRecordingIndex{mutateFilters: true}
	app := newBatchTestApp(t, embedder, index)

	_, err := app.SearchBatch(context.Background(), []string{"one", "two"}, reliquary.WithFilter(map[string]any{"tenant": "original"}))
	if err != nil {
		t.Fatalf("SearchBatch: %v", err)
	}
	if got := index.filterValues[1]; got != "original" {
		t.Fatalf("second filter tenant = %#v, want original", got)
	}
}

func TestSearchBatchClonesVectorForEveryIndexCall(t *testing.T) {
	shared := embedding.Vector{1, 1}
	embedder := &batchRecordingEmbedder{result: embedding.Result{Vectors: []embedding.Vector{shared, shared}}}
	index := &batchRecordingIndex{mutateVectors: true}
	app := newBatchTestApp(t, embedder, index)

	_, err := app.SearchBatch(context.Background(), []string{"one", "two"})
	if err != nil {
		t.Fatalf("SearchBatch: %v", err)
	}
	if got, want := index.vectorValues, []float32{1, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("index vector values = %#v, want %#v", got, want)
	}
	if shared[0] != 1 {
		t.Fatalf("embedder-owned vector mutated: %#v", shared)
	}
}

func TestSearchBatchReturnsNoPartialResults(t *testing.T) {
	want := errors.New("second search failed")
	embedder := &batchRecordingEmbedder{}
	index := &batchRecordingIndex{
		results: []*retrieval.Result{{ID: "partial", Content: "content", Embedding: []float64{1, 1}}},
		failAt:  2,
		err:     want,
	}
	app := newBatchTestApp(t, embedder, index)

	rows, err := app.SearchBatch(context.Background(), []string{"one", "two", "three"})
	if !errors.Is(err, want) || rows != nil {
		t.Fatalf("SearchBatch = %#v, %v, want nil, %v", rows, err, want)
	}
	if len(index.queries) != 2 {
		t.Fatalf("index calls = %d, want stop at 2", len(index.queries))
	}
}

func TestSearchBatchOptionsAndIndividualSearchParity(t *testing.T) {
	ctx := context.Background()
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, testDocs()...); err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	queries := []string{"go memory", "pasta dough"}
	opts := []reliquary.SearchOption{reliquary.CandidateLimit(3), reliquary.TopK(2), reliquary.WithMMR(0.4)}
	batch, err := app.SearchBatch(ctx, queries, opts...)
	if err != nil {
		t.Fatalf("SearchBatch: %v", err)
	}
	for i, query := range queries {
		individual, err := app.Search(ctx, query, opts...)
		if err != nil {
			t.Fatalf("Search %q: %v", query, err)
		}
		if !reflect.DeepEqual(batch[i], individual) {
			t.Fatalf("query %q batch = %#v, individual = %#v", query, batch[i], individual)
		}
	}
}

func TestSearchBatchDoesNotMutateIndexCandidates(t *testing.T) {
	stored := &retrieval.Result{
		ID: "stored", Content: "content", Metadata: map[string]any{"key": "value"},
		Embedding: []float64{1, 1}, CombinedScore: 42,
	}
	embedder := &batchRecordingEmbedder{}
	index := &batchRecordingIndex{results: []*retrieval.Result{stored}}
	app := newBatchTestApp(t, embedder, index)

	rows, err := app.SearchBatch(context.Background(), []string{"one", "two"})
	if err != nil {
		t.Fatalf("SearchBatch: %v", err)
	}
	rows[0][0].Metadata["key"] = "changed"
	rows[0][0].Embedding[0] = 99
	if stored.CombinedScore != 42 || stored.Metadata["key"] != "value" || stored.Embedding[0] != 1 {
		t.Fatalf("stored candidate mutated: %#v", stored)
	}
	if rows[1][0].Metadata["key"] != "value" || rows[1][0].Embedding[0] != 1 {
		t.Fatalf("later row aliases earlier row: %#v", rows[1][0])
	}
}

func TestSearchBatchCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	app := reliquary.Quickstart()
	rows, err := app.SearchBatch(ctx, []string{"one", "two"})
	if !errors.Is(err, context.Canceled) || rows != nil {
		t.Fatalf("SearchBatch canceled = %#v, %v", rows, err)
	}
}

func TestSearchBatchCancellationBetweenIndexCallsReturnsNoPartialResults(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	index := &batchRecordingIndex{cancelAt: 1, cancel: cancel}
	app := newBatchTestApp(t, &batchRecordingEmbedder{}, index)

	rows, err := app.SearchBatch(ctx, []string{"one", "two", "three"})
	if !errors.Is(err, context.Canceled) || rows != nil {
		t.Fatalf("SearchBatch canceled = %#v, %v", rows, err)
	}
	if len(index.queries) != 1 {
		t.Fatalf("index calls = %d, want 1 completed call", len(index.queries))
	}
}

func TestNilAppSearchBatchReturnsError(t *testing.T) {
	var app *reliquary.App
	if _, err := app.SearchBatch(context.Background(), []string{"query"}); !errors.Is(err, reliquary.ErrNilApp) {
		t.Fatalf("nil SearchBatch error = %v, want ErrNilApp", err)
	}
}

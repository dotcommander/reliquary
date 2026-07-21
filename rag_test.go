package reliquary_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embed"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

func testDocs() []document.Document {
	return []document.Document{
		{ID: "go-gc", Title: "go-garbage-collection.md", Format: document.FormatMarkdown, Text: "Go uses a concurrent garbage collector to reclaim unreachable memory while keeping pauses short."},
		{ID: "pasta", Title: "fresh-pasta.md", Format: document.FormatMarkdown, Text: "Fresh pasta dough rests before rolling so tagliatelle holds sauce and keeps its bite."},
		{ID: "stars", Title: "neutron-stars.md", Format: document.FormatMarkdown, Text: "Neutron stars form from collapsed stellar cores and pack enormous mass into a tiny radius."},
	}
}

func TestNewRequiresEmbedder(t *testing.T) {
	t.Parallel()
	if _, err := reliquary.New(); err == nil {
		t.Fatal("New() with no embedder = nil error, want ErrNoEmbedder")
	}
	if _, err := reliquary.New(reliquary.WithEmbedder(nil)); !errors.Is(err, reliquary.ErrNoEmbedder) {
		t.Fatalf("New(WithEmbedder(nil)) error = %v, want ErrNoEmbedder", err)
	}
	var typedNil *resultEmbedder
	if _, err := reliquary.New(reliquary.WithEmbedder(typedNil), reliquary.WithIndexIdentity("test")); !errors.Is(err, reliquary.ErrNoEmbedder) {
		t.Fatalf("New(WithEmbedder(typed nil)) error = %v, want ErrNoEmbedder", err)
	}
}

func TestNewRequiresIndexIdentity(t *testing.T) {
	t.Parallel()
	if _, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64))); !errors.Is(err, reliquary.ErrInvalidIndexIdentity) {
		t.Fatalf("New without identity error = %v", err)
	}
}

func TestNilAppReturnsErrors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	var app *reliquary.App
	if _, err := app.Ingest(ctx, testDocs()...); !errors.Is(err, reliquary.ErrNilApp) {
		t.Fatalf("nil Ingest error = %v, want ErrNilApp", err)
	}
	if _, err := app.Search(ctx, "go"); !errors.Is(err, reliquary.ErrNilApp) {
		t.Fatalf("nil Search error = %v, want ErrNilApp", err)
	}
}

func TestQuickstartIngestSearch(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	app := reliquary.Quickstart()

	n, err := app.Ingest(ctx, testDocs()...)
	if err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	if n != 3 {
		t.Fatalf("Ingest chunk count = %d, want 3", n)
	}

	hits, err := app.Search(ctx, "how does Go reclaim memory", reliquary.TopK(1), reliquary.WithMMR(0.5))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("Search returned %d hits, want 1", len(hits))
	}
	if !strings.HasPrefix(hits[0].ID, "go-gc#") {
		t.Fatalf("top hit = %q, want a go-gc chunk", hits[0].ID)
	}
}

func TestAppIndexIdentityPreventsSameDimensionMixing(t *testing.T) {
	ctx := context.Background()
	idx := reliquary.NewMemoryIndex()
	first, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("hashing-v1|smart-220-0"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Ingest(ctx, document.Document{ID: "one", Text: "identity one"}); err != nil {
		t.Fatal(err)
	}
	second, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("hashing-v2|smart-220-0"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.Ingest(ctx, document.Document{ID: "one", Text: "replacement identity"}); !errors.Is(err, reliquary.ErrIdentityMismatch) {
		t.Fatalf("Ingest mismatch error = %v", err)
	}
	if _, err := second.Search(ctx, "identity"); !errors.Is(err, reliquary.ErrIdentityMismatch) {
		t.Fatalf("Search mismatch error = %v", err)
	}
}

func TestNewWithEmbedder(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	app, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(128)), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := app.Ingest(ctx, testDocs()...); err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	hits, err := app.Search(ctx, "pasta dough", reliquary.TopK(2))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("Search returned %d hits, want 2", len(hits))
	}
}

func TestSearchEmptyStates(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	app := reliquary.Quickstart()

	hits, err := app.Search(ctx, "go")
	if err != nil {
		t.Fatalf("Search empty corpus: %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("Search empty corpus returned %d hits, want 0", len(hits))
	}

	if _, err := app.Ingest(ctx, testDocs()...); err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	hits, err = app.Search(ctx, "   ")
	if err != nil {
		t.Fatalf("Search empty query: %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("Search empty query returned %d hits, want 0", len(hits))
	}
}

func TestEmptyQueryDoesNotInvokeIndex(t *testing.T) {
	t.Parallel()
	idx := &recordingIndex{}
	app, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := app.Search(context.Background(), " \t\n "); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if idx.searchCalls != 0 {
		t.Fatalf("index Search calls = %d, want 0", idx.searchCalls)
	}
}

func TestCandidateLimitIsDistinctFromTopKAndMMR(t *testing.T) {
	t.Parallel()
	idx := &recordingIndex{results: []*retrieval.Result{
		{ID: "a", Content: "go memory", Embedding: []float64{1, 0}},
		{ID: "b", Content: "go memory", Embedding: []float64{0.99, 0.1}},
		{ID: "c", Content: "go memory", Embedding: []float64{0, 1}},
	}}
	app, err := reliquary.New(reliquary.WithEmbedder(fixedEmbedder{vector: embeddings.Vector{1, 0}}), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	hits, err := app.Search(context.Background(), "go memory", reliquary.CandidateLimit(3), reliquary.TopK(2), reliquary.WithMMR(0))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if idx.lastQuery.Limit != 3 {
		t.Fatalf("candidate limit = %d, want 3", idx.lastQuery.Limit)
	}
	if len(hits) != 2 || hits[0].ID != "a" || hits[1].ID != "c" {
		t.Fatalf("MMR results = %#v, want IDs [a c]", hits)
	}
}

func TestWithFilterPassesSnapshotToIndex(t *testing.T) {
	t.Parallel()
	idx := &recordingIndex{results: []*retrieval.Result{
		{ID: "a", Content: "go memory", Embedding: []float64{1, 0}},
	}}
	app, err := reliquary.New(reliquary.WithEmbedder(fixedEmbedder{vector: embeddings.Vector{1, 0}}), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	filter := map[string]any{"project": "example", "archived": false}
	option := reliquary.WithFilter(filter)
	filter["project"] = "changed"
	filter["added"] = true

	if _, err := app.Search(context.Background(), "go memory", option, reliquary.TopK(1)); err != nil {
		t.Fatalf("Search: %v", err)
	}
	want := map[string]any{"project": "example", "archived": false}
	if !reflect.DeepEqual(idx.lastQuery.Filter, want) {
		t.Fatalf("index filter = %#v, want %#v", idx.lastQuery.Filter, want)
	}
}

func TestWithFilterRejectsCompoundValueBeforeIndexSearch(t *testing.T) {
	t.Parallel()
	idx := &recordingIndex{}
	app, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = app.Search(context.Background(), "go", reliquary.WithFilter(map[string]any{"tenant": []string{"one"}}))
	if err == nil || !strings.Contains(err.Error(), `filter "tenant" must be scalar`) {
		t.Fatalf("Search error = %v, want scalar filter error", err)
	}
	if idx.searchCalls != 0 {
		t.Fatalf("index Search calls = %d, want 0", idx.searchCalls)
	}
}

func TestWithFilterRejectsNonFiniteNumberBeforeEmbedding(t *testing.T) {
	t.Parallel()
	embedErr := errors.New("embedder must not be called")
	idx := &recordingIndex{}
	app, err := reliquary.New(reliquary.WithEmbedder(errEmbedder{err: embedErr}), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = app.Search(context.Background(), "go", reliquary.WithFilter(map[string]any{"score": math.NaN()}))
	if err == nil || errors.Is(err, embedErr) || !strings.Contains(err.Error(), "must be finite") {
		t.Fatalf("Search error = %v, want pre-embedding finite-number error", err)
	}
	if idx.searchCalls != 0 {
		t.Fatalf("index Search calls = %d, want 0", idx.searchCalls)
	}
}

func TestWithIndexNilFallsBackToDefault(t *testing.T) {
	t.Parallel()
	var typedNil *recordingIndex
	for _, tt := range []struct {
		name  string
		index reliquary.Index
	}{
		{name: "nil"},
		{name: "typed nil", index: typedNil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			app, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(tt.index), reliquary.WithIndexIdentity("test"))
			if err != nil {
				t.Fatalf("New with default fallback: %v", err)
			}
			if hits, err := app.Search(context.Background(), "go"); err != nil || len(hits) != 0 {
				t.Fatalf("default fallback Search = %#v, %v", hits, err)
			}
		})
	}
}

func TestIndexErrorPropagatesWithoutResults(t *testing.T) {
	t.Parallel()
	want := errors.New("index failed")
	idx := &recordingIndex{searchErr: want, results: []*retrieval.Result{{ID: "partial"}}}
	app, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	hits, err := app.Search(context.Background(), "go")
	if !errors.Is(err, want) {
		t.Fatalf("Search error = %v, want %v", err, want)
	}
	if hits != nil {
		t.Fatalf("Search hits = %#v, want nil", hits)
	}
}

func TestSearchDoesNotMutateIndexCandidates(t *testing.T) {
	t.Parallel()
	stored := &retrieval.Result{ID: "stored", Content: "go memory", Embedding: []float64{1, 0}, CombinedScore: 42}
	idx := &recordingIndex{results: []*retrieval.Result{stored}}
	app, err := reliquary.New(reliquary.WithEmbedder(fixedEmbedder{vector: embeddings.Vector{1, 0}}), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := app.Search(context.Background(), "go"); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if stored.CombinedScore != 42 {
		t.Fatalf("stored CombinedScore = %v, want 42", stored.CombinedScore)
	}
}

func TestTopKBounds(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, testDocs()...); err != nil {
		t.Fatalf("Ingest: %v", err)
	}

	for _, k := range []int{0, -3} {
		t.Run(fmt.Sprintf("k=%d", k), func(t *testing.T) {
			hits, err := app.Search(ctx, "go memory", reliquary.TopK(k))
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if len(hits) != 0 {
				t.Fatalf("TopK(%d) returned %d hits, want 0", k, len(hits))
			}
		})
	}

	hits, err := app.Search(ctx, "go memory", reliquary.TopK(99))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != len(testDocs()) {
		t.Fatalf("TopK over limit returned %d hits, want %d", len(hits), len(testDocs()))
	}
}

func TestMMRLambdaBounds(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, testDocs()...); err != nil {
		t.Fatalf("Ingest: %v", err)
	}

	for _, lambda := range []float64{-10, 0, 1, 10} {
		t.Run(fmt.Sprintf("lambda=%g", lambda), func(t *testing.T) {
			hits, err := app.Search(ctx, "go memory", reliquary.TopK(2), reliquary.WithMMR(lambda))
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if len(hits) != 2 {
				t.Fatalf("WithMMR(%g) returned %d hits, want 2", lambda, len(hits))
			}
		})
	}
}

func TestContextCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, testDocs()...); !errors.Is(err, context.Canceled) {
		t.Fatalf("Ingest canceled error = %v, want context.Canceled", err)
	}
	if _, err := app.Search(ctx, "go"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Search canceled error = %v, want context.Canceled", err)
	}
}

func TestIngestRejectsInvalidDocumentIDsBeforeEmbedding(t *testing.T) {
	t.Parallel()
	embedErr := errors.New("embedder must not be called")
	app, err := reliquary.New(reliquary.WithEmbedder(errEmbedder{err: embedErr}), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := app.Ingest(context.Background(), document.Document{ID: " \t", Text: "content"}); !errors.Is(err, reliquary.ErrInvalidDocumentID) {
		t.Fatalf("blank ID error = %v, want ErrInvalidDocumentID", err)
	}
	if _, err := app.Ingest(context.Background(), document.Document{ID: "same", Text: "one"}, document.Document{ID: "same", Text: "two"}); !errors.Is(err, reliquary.ErrDuplicateDocumentID) {
		t.Fatalf("duplicate ID error = %v, want ErrDuplicateDocumentID", err)
	}
}

func TestIngestWithNoDocumentsDoesNotWriteIndex(t *testing.T) {
	t.Parallel()
	idx := &recordingIndex{upsertErr: errors.New("index must not be called")}
	app, err := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(16)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if n, err := app.Ingest(context.Background()); err != nil || n != 0 {
		t.Fatalf("Ingest() = %d, %v, want 0, nil", n, err)
	}
	if idx.replaceCalls != 0 {
		t.Fatalf("ReplaceDocuments calls = %d, want 0", idx.replaceCalls)
	}
}

func TestIngestReplacesCompleteDocumentRevision(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	idx := reliquary.NewMemoryIndex()
	app, err := reliquary.New(
		reliquary.WithEmbedder(embed.NewHashing(16)),
		reliquary.WithIndex(idx),
		reliquary.WithIndexIdentity("test"),
		reliquary.WithChunker(chunking.SmartBoundary, 20, 0),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if n, err := app.Ingest(ctx, document.Document{ID: "doc", Text: strings.Repeat("alpha beta gamma ", 8)}); err != nil || n < 2 {
		t.Fatalf("initial Ingest = %d, %v, want multiple chunks", n, err)
	}
	if n, err := app.Ingest(ctx, document.Document{ID: "doc", Text: "short revision"}); err != nil || n != 1 {
		t.Fatalf("short Ingest = %d, %v, want one chunk", n, err)
	}
	got, err := idx.Search(ctx, reliquary.IndexQuery{Identity: "test", Filter: map[string]any{"document_id": "doc"}})
	if err != nil || len(got) != 1 || got[0].Content != "short revision" {
		t.Fatalf("after short revision = %#v, %v", got, err)
	}
	if n, err := app.Ingest(ctx, document.Document{ID: "doc", Text: ""}); err != nil || n != 0 {
		t.Fatalf("empty Ingest = %d, %v", n, err)
	}
	got, err = idx.Search(ctx, reliquary.IndexQuery{Identity: "test", Filter: map[string]any{"document_id": "doc"}})
	if err != nil || len(got) != 0 {
		t.Fatalf("after empty revision = %#v, %v", got, err)
	}
}

func TestErrorPropagation(t *testing.T) {
	t.Parallel()
	embedErr := errors.New("embed failed")
	indexErr := errors.New("index failed")
	ctx := context.Background()

	app, err := reliquary.New(reliquary.WithEmbedder(errEmbedder{err: embedErr}), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := app.Ingest(ctx, testDocs()...); !errors.Is(err, embedErr) {
		t.Fatalf("Ingest embed error = %v, want %v", err, embedErr)
	}
	if _, err := app.Search(ctx, "go"); !errors.Is(err, embedErr) {
		t.Fatalf("Search embed error = %v, want %v", err, embedErr)
	}

	app, err = reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(&recordingIndex{searchErr: indexErr, upsertErr: indexErr}), reliquary.WithIndexIdentity("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := app.Ingest(ctx, testDocs()...); !errors.Is(err, indexErr) {
		t.Fatalf("Ingest index error = %v, want %v", err, indexErr)
	}
	if _, err := app.Search(ctx, "go"); !errors.Is(err, indexErr) {
		t.Fatalf("Search index error = %v, want %v", err, indexErr)
	}
}

func TestIngestRejectsMalformedEmbeddingBeforeIndexMutation(t *testing.T) {
	t.Parallel()

	idx := &recordingIndex{}
	app, err := reliquary.New(
		reliquary.WithEmbedder(resultEmbedder{result: embeddings.Result{}}),
		reliquary.WithIndex(idx),
		reliquary.WithIndexIdentity("test"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = app.Ingest(context.Background(), document.Document{ID: "doc", Text: "content"})
	if !errors.Is(err, embeddings.ErrInvalidResult) || !errors.Is(err, retrieval.ErrEmbeddingCountMismatch) {
		t.Fatalf("Ingest error = %v, want ErrInvalidResult and ErrEmbeddingCountMismatch", err)
	}
	if idx.replaceCalls != 0 {
		t.Fatalf("ReplaceDocuments calls = %d, want 0", idx.replaceCalls)
	}
}

func TestSearchRejectsMalformedEmbeddingBeforeIndexAccess(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		result embeddings.Result
	}{
		{name: "count mismatch", result: embeddings.Result{}},
		{name: "empty vector", result: embeddings.Result{Vectors: []embeddings.Vector{{}}}},
		{name: "declared dimension mismatch", result: embeddings.Result{Model: embeddings.ModelRef{Dim: 3}, Vectors: []embeddings.Vector{{1, 0}}}},
		{name: "non-finite vector", result: embeddings.Result{Vectors: []embeddings.Vector{{float32(math.Inf(1)), 0}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			idx := &recordingIndex{}
			app, err := reliquary.New(
				reliquary.WithEmbedder(resultEmbedder{result: tt.result}),
				reliquary.WithIndex(idx),
				reliquary.WithIndexIdentity("test"),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := app.Search(context.Background(), "query"); !errors.Is(err, embeddings.ErrInvalidResult) {
				t.Fatalf("Search error = %v, want ErrInvalidResult", err)
			}
			if idx.searchCalls != 0 {
				t.Fatalf("Index.Search calls = %d, want 0", idx.searchCalls)
			}
		})
	}
}

func TestInvalidChunkerPropagates(t *testing.T) {
	t.Parallel()
	app, err := reliquary.New(
		reliquary.WithEmbedder(embed.NewHashing(64)),
		reliquary.WithIndexIdentity("test"),
		reliquary.WithChunker(chunking.Strategy("nope"), 220, 0),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := app.Ingest(context.Background(), testDocs()...); !errors.Is(err, chunking.ErrUnknownStrategy) {
		t.Fatalf("Ingest invalid chunker error = %v, want ErrUnknownStrategy", err)
	}
}

func TestResetIndexAllowsIdentityRebuild(t *testing.T) {
	ctx := context.Background()
	idx := reliquary.NewMemoryIndex()
	first, _ := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("old"))
	if _, err := first.Ingest(ctx, document.Document{ID: "old", Text: "old data"}); err != nil {
		t.Fatal(err)
	}
	if err := first.ResetIndex(ctx); err != nil {
		t.Fatal(err)
	}
	second, _ := reliquary.New(reliquary.WithEmbedder(embed.NewHashing(64)), reliquary.WithIndex(idx), reliquary.WithIndexIdentity("new"))
	if _, err := second.Ingest(ctx, document.Document{ID: "new", Text: "new data"}); err != nil {
		t.Fatal(err)
	}
}

type errEmbedder struct {
	err error
}

type fixedEmbedder struct{ vector embeddings.Vector }

type resultEmbedder struct {
	result embeddings.Result
}

func (e resultEmbedder) Embed(context.Context, embeddings.Request) (embeddings.Result, error) {
	return e.result, nil
}

func (e fixedEmbedder) Embed(context.Context, embeddings.Request) (embeddings.Result, error) {
	return embeddings.Result{Vectors: []embeddings.Vector{e.vector}}, nil
}

type recordingIndex struct {
	searchCalls  int
	replaceCalls int
	lastQuery    reliquary.IndexQuery
	results      []*retrieval.Result
	searchErr    error
	upsertErr    error
}

func (i *recordingIndex) Upsert(context.Context, []*retrieval.Result) error { return i.upsertErr }
func (i *recordingIndex) ReplaceDocuments(context.Context, []reliquary.DocumentReplacement) error {
	i.replaceCalls++
	return i.upsertErr
}
func (i *recordingIndex) DeleteDocument(context.Context, string) error { return nil }
func (i *recordingIndex) Search(_ context.Context, query reliquary.IndexQuery) ([]*retrieval.Result, error) {
	i.searchCalls++
	i.lastQuery = query
	return i.results, i.searchErr
}

func (e errEmbedder) Embed(context.Context, embeddings.Request) (embeddings.Result, error) {
	return embeddings.Result{}, e.err
}

func TestSearchRepeatable(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, testDocs()...); err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	first, err := app.Search(ctx, "garbage collector memory", reliquary.TopK(3))
	if err != nil {
		t.Fatalf("Search 1: %v", err)
	}
	second, err := app.Search(ctx, "garbage collector memory", reliquary.TopK(3))
	if err != nil {
		t.Fatalf("Search 2: %v", err)
	}
	if len(first) != len(second) {
		t.Fatalf("result count drift: %d then %d", len(first), len(second))
	}
	for i := range first {
		if first[i].ID != second[i].ID || first[i].CombinedScore != second[i].CombinedScore {
			t.Fatalf("search not repeatable at %d: %q/%.4f vs %q/%.4f", i, first[i].ID, first[i].CombinedScore, second[i].ID, second[i].CombinedScore)
		}
	}
}

func TestSearchConcurrent(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	app := reliquary.Quickstart()
	if _, err := app.Ingest(ctx, testDocs()...); err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := app.Search(ctx, "concurrent memory query", reliquary.TopK(2)); err != nil {
				t.Errorf("Search: %v", err)
			}
		}()
	}
	wg.Wait()
}

func TestConcurrentIngestSearch(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	app := reliquary.Quickstart()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := app.Ingest(ctx, testDocs()...); err != nil {
				t.Errorf("Ingest: %v", err)
			}
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := app.Search(ctx, "concurrent ingest and search", reliquary.TopK(2)); err != nil {
				t.Errorf("Search: %v", err)
			}
		}()
	}
	wg.Wait()
}

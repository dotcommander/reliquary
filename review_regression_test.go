package reliquary_test

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embed"
	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestNewRejectsInvalidChunkerSize(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name    string
		size    int
		overlap int
		want    error
	}{
		{"zero size", 0, 0, reliquary.ErrInvalidChunkSize},
		{"negative size", -1, 0, reliquary.ErrInvalidChunkSize},
		{"negative overlap", 100, -1, reliquary.ErrInvalidChunkOverlap},
		{"overlap equals size", 100, 100, reliquary.ErrInvalidChunkOverlap},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := reliquary.New(
				reliquary.WithEmbedder(embed.NewHashing(8)),
				reliquary.WithIndexIdentity("test"),
				reliquary.WithChunker(chunking.SmartBoundary, tc.size, tc.overlap),
			)
			if !errors.Is(err, tc.want) {
				t.Fatalf("New error = %v, want %v", err, tc.want)
			}
		})
	}
}

type cancelingIndex struct {
	reliquary.Index
	cancel context.CancelFunc
	items  []*retrieval.Result
}

func (i *cancelingIndex) Search(context.Context, reliquary.IndexQuery) ([]*retrieval.Result, error) {
	i.cancel()
	return i.items, nil
}

func TestSearchReturnsContextErrorWhenCanceledAfterIndex(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	idx := &cancelingIndex{
		cancel: cancel,
		items: []*retrieval.Result{
			{ID: "a", Content: "hello world", Embedding: []float64{1, 0}},
		},
	}
	app, err := reliquary.New(
		reliquary.WithEmbedder(embed.NewHashing(2)),
		reliquary.WithIndex(idx),
		reliquary.WithIndexIdentity("test"),
	)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := app.Search(ctx, "hello")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Search error = %v, want context.Canceled", err)
	}
	if rows != nil {
		t.Fatalf("Search rows = %v, want nil", rows)
	}
}

func TestAppIngestEmptyDocRespectsIndexIdentity(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	sharedIdx := reliquary.NewMemoryIndex()
	embedder := embed.NewHashing(16)

	appA, err := reliquary.New(
		reliquary.WithEmbedder(embedder),
		reliquary.WithIndex(sharedIdx),
		reliquary.WithIndexIdentity("space-a"),
	)
	if err != nil {
		t.Fatalf("New(appA): %v", err)
	}
	appB, err := reliquary.New(
		reliquary.WithEmbedder(embedder),
		reliquary.WithIndex(sharedIdx),
		reliquary.WithIndexIdentity("space-b"),
	)
	if err != nil {
		t.Fatalf("New(appB): %v", err)
	}

	if _, err := appA.Ingest(ctx, document.Document{ID: "doc-1", Text: "alpha beta gamma"}); err != nil {
		t.Fatalf("appA.Ingest: %v", err)
	}

	// Empty content produces zero chunks; must still fail identity check and not delete appA's doc.
	_, err = appB.Ingest(ctx, document.Document{ID: "doc-1", Text: "   "})
	if !errors.Is(err, indexcontract.ErrIdentityMismatch) {
		t.Fatalf("appB.Ingest(empty) error = %v, want ErrIdentityMismatch", err)
	}

	results, err := appA.Search(ctx, "alpha")
	if err != nil {
		t.Fatalf("appA.Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("appA document was deleted by mismatched identity empty ingest")
	}
}

func TestAppSearchValidatesOptionsOnBlankQueries(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	app := reliquary.InMemory(16)

	invalidFilter := reliquary.WithFilter(map[string]any{"nested": map[string]any{"bad": true}})

	if _, err := app.Search(ctx, "   ", invalidFilter); err == nil {
		t.Fatal("Search(blank, invalidFilter) error = nil, want validation error")
	}
	if _, err := app.SearchBatch(ctx, []string{"", " \t "}, invalidFilter); err == nil {
		t.Fatal("SearchBatch(all blank, invalidFilter) error = nil, want validation error")
	}
}

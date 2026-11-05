package reliquary_test

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/embed"
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

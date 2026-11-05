package sqlite

import (
	"context"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestSearchSkipsEmbeddingDecodeForFilterRejectedRows(t *testing.T) {
	t.Parallel()
	db := openTestDB(t)
	idx, _ := New(db, Config{})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{
		{ID: "rejected", Content: "needle rejected", Embedding: []float64{1, 0}, Metadata: map[string]any{"value": uint64(1)}},
		{ID: "kept", Content: "needle kept", Embedding: []float64{1, 0}, Metadata: map[string]any{"value": uint64(2)}},
	}); err != nil {
		t.Fatal(err)
	}
	// Corrupt the rejected row's embedding payload. A correct implementation
	// filters the row out by metadata before decoding its embedding, so the
	// invalid JSON must never be parsed.
	if _, err := db.ExecContext(context.Background(), `UPDATE reliquary_index SET embedding = 'not-json' WHERE id = 'rejected'`); err != nil {
		t.Fatal(err)
	}
	got, err := idx.Search(context.Background(), indexcontract.IndexQuery{
		Text:   "needle",
		Filter: map[string]any{"value": uint64(2)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "kept" {
		t.Fatalf("filtered candidates = %#v", got)
	}
}

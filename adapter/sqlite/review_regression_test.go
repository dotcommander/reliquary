package sqlite

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"testing"
	"time"

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

func TestSearchNegativeLimitBoundedByCandidateLimit(t *testing.T) {
	t.Parallel()
	db := openTestDB(t)
	idx, err := New(db, Config{CandidateLimit: 2})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if err := idx.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	items := []*retrieval.Result{
		{ID: "a", Content: "alpha"},
		{ID: "b", Content: "beta"},
		{ID: "c", Content: "gamma"},
		{ID: "d", Content: "delta"},
		{ID: "e", Content: "epsilon"},
	}
	if err := idx.Upsert(ctx, items); err != nil {
		t.Fatal(err)
	}
	got, err := idx.searchItems(ctx, indexcontract.IndexQuery{Limit: -5})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("searchItems(Limit=-5) returned %d candidates, want 2", len(got))
	}
}

func TestWithTxIgnoresErrTxDoneOnContextCancel(t *testing.T) {
	t.Parallel()
	db := openTestDB(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := withTx(ctx, db, func(tx *sql.Tx) error {
		cancel()
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			if _, execErr := tx.ExecContext(context.Background(), `SELECT 1`); errors.Is(execErr, sql.ErrTxDone) {
				break
			}
			time.Sleep(time.Millisecond)
		}
		return ctx.Err()
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("withTx error = %v, want context.Canceled", err)
	}
	if errors.Is(err, sql.ErrTxDone) || strings.Contains(err.Error(), "rollback failed") {
		t.Fatalf("withTx wrapped ErrTxDone as rollback failure: %v", err)
	}
}

func TestReservedKeywordTableNames(t *testing.T) {
	t.Parallel()
	for _, table := range []string{"order", "group"} {
		t.Run(table, func(t *testing.T) {
			t.Parallel()
			db := openTestDB(t)
			ctx := context.Background()
			idx, err := New(db, Config{Table: table})
			if err != nil {
				t.Fatalf("New(%q): %v", table, err)
			}
			if err := idx.Migrate(ctx); err != nil {
				t.Fatalf("Migrate(%q): %v", table, err)
			}
			if err := idx.Upsert(ctx, []*retrieval.Result{
				{ID: "r1", DocumentID: "doc1", Filename: "a.txt", Content: "quartz beacon", Metadata: map[string]any{"tag": "v1"}},
				{ID: "r2", DocumentID: "doc2", Filename: "b.txt", Content: "other content", Metadata: map[string]any{"tag": "v2"}},
			}); err != nil {
				t.Fatalf("Upsert(%q): %v", table, err)
			}
			got, err := idx.Search(ctx, indexcontract.IndexQuery{Text: "quartz", Filter: map[string]any{"tag": "v1"}, Limit: 5})
			if err != nil {
				t.Fatalf("Search(%q): %v", table, err)
			}
			if len(got) != 1 || got[0].ID != "r1" {
				t.Fatalf("Search(%q) = %#v, want [r1]", table, got)
			}
			if err := idx.ReplaceDocuments(ctx, []indexcontract.DocumentReplacement{
				{DocumentID: "doc1", Results: []*retrieval.Result{
					{ID: "r1", DocumentID: "doc1", Filename: "a.txt", Content: "amber lantern", Metadata: map[string]any{"tag": "v1"}},
				}},
			}); err != nil {
				t.Fatalf("ReplaceDocuments(%q): %v", table, err)
			}
			got, err = idx.Search(ctx, indexcontract.IndexQuery{Text: "amber", Limit: 5})
			if err != nil {
				t.Fatalf("Search after ReplaceDocuments(%q): %v", table, err)
			}
			if len(got) != 1 || got[0].ID != "r1" {
				t.Fatalf("Search after ReplaceDocuments(%q) = %#v, want [r1]", table, got)
			}
			if err := idx.RepairFTS(ctx); err != nil {
				t.Fatalf("RepairFTS(%q): %v", table, err)
			}
			if err := idx.DeleteDocument(ctx, "doc2"); err != nil {
				t.Fatalf("DeleteDocument(%q): %v", table, err)
			}
			if err := idx.Reset(ctx); err != nil {
				t.Fatalf("Reset(%q): %v", table, err)
			}
			got, err = idx.Search(ctx, indexcontract.IndexQuery{Limit: 5})
			if err != nil {
				t.Fatalf("Search after Reset(%q): %v", table, err)
			}
			if len(got) != 0 {
				t.Fatalf("Search after Reset(%q) = %#v, want empty", table, got)
			}
		})
	}
}

package sqlite

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestReplacementLookupBatchesAndPreservesAtomicFTS(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	db := openTestDB(t)
	idx, err := New(db, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	retained := []*retrieval.Result{
		{ID: "conflict-early", DocumentID: "retained", Content: "retained early"},
		{ID: "conflict-late", DocumentID: "retained", Content: "retained late"},
	}
	if err := idx.Upsert(ctx, retained); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(ctx, []*retrieval.Result{{ID: "old", DocumentID: "revised", Content: "obsolete term"}}); err != nil {
		t.Fatal(err)
	}
	items := make([]*retrieval.Result, 260)
	for n := range items {
		items[n] = &retrieval.Result{ID: fmt.Sprintf("new-%03d", n), DocumentID: "revised", Content: "fresh term"}
	}
	items[12].ID = "conflict-early"
	items[258].ID = "conflict-late"
	revision := []indexcontract.DocumentReplacement{{DocumentID: "revised", Results: items}}
	err = idx.ReplaceDocuments(ctx, revision)
	if !errors.Is(err, indexcontract.ErrResultIDConflict) || !strings.Contains(err.Error(), `"conflict-early"`) {
		t.Fatalf("first conflict = %v", err)
	}
	var count int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index WHERE id = 'old'`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("old row after rollback = %d, %v", count, err)
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index_fts WHERE id = 'old'`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("old FTS row after rollback = %d, %v", count, err)
	}
	items[12].ID = "new-012"
	items[258].ID = "new-258"
	// A stale FTS row without a base row must also be removed before insertion.
	if _, err := db.ExecContext(ctx, `INSERT INTO reliquary_index_fts (id, content, filename) VALUES ('new-012', 'orphan term', '')`); err != nil {
		t.Fatal(err)
	}
	if err := idx.ReplaceDocuments(ctx, revision); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index WHERE document_id = 'revised'`).Scan(&count); err != nil || count != len(items) {
		t.Fatalf("new base rows = %d, %v", count, err)
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index_fts WHERE content = 'fresh term'`).Scan(&count); err != nil || count != len(items) {
		t.Fatalf("new FTS rows = %d, %v", count, err)
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index_fts WHERE id = 'old'`).Scan(&count); err != nil || count != 0 {
		t.Fatalf("stale FTS rows = %d, %v", count, err)
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index_fts WHERE id = 'new-012' AND content = 'fresh term'`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("replaced orphan FTS rows = %d, %v", count, err)
	}
}

func TestReplacementPreservesReassignedUpsertFTS(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	db := openTestDB(t)
	idx, err := New(db, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(ctx, []*retrieval.Result{{ID: "shared", DocumentID: "old-doc", Content: "old content"}}); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(ctx, []*retrieval.Result{{ID: "shared", DocumentID: "new-doc", Content: "reassigned content"}}); err != nil {
		t.Fatal(err)
	}
	if err := idx.ReplaceDocuments(ctx, []indexcontract.DocumentReplacement{{DocumentID: "old-doc", Results: []*retrieval.Result{{ID: "shared", DocumentID: "old-doc"}}}}); !errors.Is(err, indexcontract.ErrResultIDConflict) {
		t.Fatalf("reassigned ID conflict = %v", err)
	}
	if err := idx.ReplaceDocuments(ctx, []indexcontract.DocumentReplacement{{DocumentID: "old-doc", Results: []*retrieval.Result{{ID: "replacement", DocumentID: "old-doc", Content: "replacement content"}}}}); err != nil {
		t.Fatal(err)
	}
	var documentID, content string
	if err := db.QueryRowContext(ctx, `SELECT document_id, content FROM reliquary_index WHERE id = 'shared'`).Scan(&documentID, &content); err != nil || documentID != "new-doc" || content != "reassigned content" {
		t.Fatalf("reassigned base row = (%q, %q), %v", documentID, content, err)
	}
	var count int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index_fts WHERE id = 'shared' AND content = 'reassigned content'`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("reassigned FTS rows = %d, %v", count, err)
	}
}

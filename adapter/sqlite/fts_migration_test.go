package sqlite

import (
	"errors"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestRepeatMigrateSkipsCurrentFTSAndExplicitRepairRestoresParity(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	db := openTestDB(t)
	idx, err := New(db, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(ctx, []*retrieval.Result{{ID: "row", Content: "current searchable", IndexIdentity: "space", Embedding: []float64{1, 0}}}); err != nil {
		t.Fatal(err)
	}
	// Simulate an external FTS write. The adapter's version marker cannot
	// observe it; repeat migration should not silently rewrite current FTS.
	if _, err := db.ExecContext(ctx, `UPDATE reliquary_index_fts SET content = 'stale searchable' WHERE id = 'row'`); err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	var content string
	if err := db.QueryRowContext(ctx, `SELECT content FROM reliquary_index_fts WHERE id = 'row'`).Scan(&content); err != nil || content != "stale searchable" {
		t.Fatalf("repeat migration FTS = %q, err = %v", content, err)
	}
	if err := idx.RepairFTS(ctx); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRowContext(ctx, `SELECT content FROM reliquary_index_fts WHERE id = 'row'`).Scan(&content); err != nil || content != "current searchable" {
		t.Fatalf("repaired FTS = %q, err = %v", content, err)
	}
	// A current FTS generation must not suppress existing identity drift checks.
	if _, err := db.ExecContext(ctx, `UPDATE reliquary_index SET index_identity = 'drift' WHERE id = 'row'`); err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
		t.Fatalf("repeat migration identity drift = %v", err)
	}
}

func TestMigrateRestoresMissingFTSDespiteCurrentVersion(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	db := openTestDB(t)
	idx, err := New(db, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(ctx, []*retrieval.Result{{ID: "row", Content: "searchable"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `DROP TABLE reliquary_index_fts`); err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	var count int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index_fts WHERE id = 'row'`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("restored FTS count = %d, err = %v", count, err)
	}
}

func TestMigrateUpgradesLegacyStateVersionAndRepairsFTS(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	db := openTestDB(t)
	for _, statement := range []string{
		`CREATE TABLE reliquary_index (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL, index_identity TEXT NOT NULL DEFAULT '')`,
		`CREATE VIRTUAL TABLE reliquary_index_fts USING fts5(id UNINDEXED, content, filename)`,
		`CREATE TABLE reliquary_index_state (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), index_identity TEXT, embedding_dimension INTEGER CHECK (embedding_dimension IS NULL OR embedding_dimension > 0))`,
		`INSERT INTO reliquary_index VALUES ('row', '', '', 'current searchable', json_object(), json_array(), '')`,
		`INSERT INTO reliquary_index_fts VALUES ('orphan', 'stale searchable', '')`,
	} {
		if _, err := db.ExecContext(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	idx, err := New(db, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	var version, count int
	if err := db.QueryRowContext(ctx, `SELECT fts_version FROM reliquary_index_state WHERE singleton = 1`).Scan(&version); err != nil || version != currentFTSVersion {
		t.Fatalf("FTS version = %d, err = %v", version, err)
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index_fts WHERE id = 'row' AND content = 'current searchable'`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("repaired FTS row count = %d, err = %v", count, err)
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM reliquary_index_fts WHERE id = 'orphan'`).Scan(&count); err != nil || count != 0 {
		t.Fatalf("orphan FTS row count = %d, err = %v", count, err)
	}
}

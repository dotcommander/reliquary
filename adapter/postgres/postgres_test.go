package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/index/indextest"
	"github.com/dotcommander/reliquary/retrieval"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

var postgresTableSequence atomic.Uint64

func TestNewValidatesConfigurationWithoutMigrating(t *testing.T) {
	t.Parallel()

	if _, err := New(nil, Config{}); err == nil {
		t.Fatal("New(nil) succeeded")
	}
	pool := &pgxpool.Pool{}
	for _, table := range []string{"bad.table", "has-hyphen", "1leading", strings.Repeat("a", 64)} {
		if _, err := New(pool, Config{Table: table}); err == nil {
			t.Fatalf("New(table=%q) succeeded", table)
		}
	}
	idx, err := New(pool, Config{})
	if err != nil {
		t.Fatalf("New(default): %v", err)
	}
	if idx.table != defaultTable {
		t.Fatalf("table = %q, want %q", idx.table, defaultTable)
	}
}

func TestSearchQueryUsesBoundedPgvectorAndDeterministicFilters(t *testing.T) {
	t.Parallel()

	idx := &Index{quoted: `"items"`}
	statement, args, err := idx.searchQuery(indexcontract.IndexQuery{
		Vector: []float32{1, 0}, Limit: 5,
		Filter: map[string]any{"tenant": "one", "document_id": "doc"},
	})
	if err != nil {
		t.Fatalf("searchQuery: %v", err)
	}
	for _, want := range []string{
		`FROM "items"`, `document_id = $1`, `metadata @> $2::jsonb`,
		`embedding <=> $3 NULLS LAST, id ASC`, `LIMIT $4`,
	} {
		if !strings.Contains(statement, want) {
			t.Fatalf("query %q does not contain %q", statement, want)
		}
	}
	if len(args) != 4 || args[3] != 5 {
		t.Fatalf("args = %#v", args)
	}
}

func TestSearchQueryPreservesExactJSONNumber(t *testing.T) {
	t.Parallel()

	idx := &Index{quoted: `"items"`}
	_, args, err := idx.searchQuery(indexcontract.IndexQuery{
		Filter: map[string]any{"large": json.Number("9007199254740993")},
	})
	if err != nil {
		t.Fatalf("searchQuery: %v", err)
	}
	if len(args) != 1 {
		t.Fatalf("args = %#v, want one JSON argument", args)
	}
	encoded, ok := args[0].([]byte)
	if !ok {
		t.Fatalf("filter argument type = %T, want []byte", args[0])
	}
	if got, want := string(encoded), `{"large":9007199254740993}`; got != want {
		t.Fatalf("encoded filter = %q, want %q", got, want)
	}
}

func TestSearchQueryRejectsStructuredMetadataFilter(t *testing.T) {
	t.Parallel()

	idx := &Index{quoted: `"items"`}
	_, _, err := idx.searchQuery(indexcontract.IndexQuery{Filter: map[string]any{"tags": []string{"go"}}})
	if err == nil {
		t.Fatal("structured filter succeeded")
	}
}

func TestIndexContract(t *testing.T) {
	ctx := context.Background()
	pool := openPostgresTestPool(t, ctx)

	var sequence atomic.Uint32
	tables := make([]string, 0, 8)
	t.Cleanup(func() {
		for _, table := range tables {
			_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, (pgx.Identifier{table}).Sanitize()))
			_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, (pgx.Identifier{boundedStateTableName(table)}).Sanitize()))
		}
	})
	indextest.Run(t, func() indexcontract.Index {
		table := fmt.Sprintf("reliquary_index_test_%d", sequence.Add(1))
		tables = append(tables, table)
		idx, err := New(pool, Config{Table: table})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if err := idx.Migrate(ctx); err != nil {
			t.Fatalf("Migrate: %v", err)
		}
		return idx
	})
}

func TestConcurrentUpsertsSerializeDimensionDiscovery(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	pool := openPostgresTestPool(t, ctx)
	table := fmt.Sprintf("reliquary_dimension_test_%d_%d", os.Getpid(), postgresTableSequence.Add(1))
	idx, err := New(pool, Config{Table: table})
	if err != nil {
		cancel()
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, idx.quoted))
		_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, idx.stateQuoted))
	})
	if err := idx.Migrate(ctx); err != nil {
		cancel()
		t.Fatalf("Migrate: %v", err)
	}

	blocker, err := pool.Begin(ctx)
	if err != nil {
		cancel()
		t.Fatalf("begin blocker: %v", err)
	}
	if _, err := blocker.Exec(ctx, fmt.Sprintf(`LOCK TABLE %s IN SHARE UPDATE EXCLUSIVE MODE`, idx.quoted)); err != nil {
		cancel()
		_ = blocker.Rollback(context.Background())
		t.Fatalf("lock blocker: %v", err)
	}

	var workers sync.WaitGroup
	results := make(chan error, 2)
	t.Cleanup(func() {
		cancel()
		_ = blocker.Rollback(context.Background())
		workers.Wait()
	})
	workers.Go(func() {
		results <- idx.Upsert(ctx, []*retrieval.Result{{ID: "two", IndexIdentity: "shared", Embedding: []float64{1, 0}}})
	})
	workers.Go(func() {
		results <- idx.Upsert(ctx, []*retrieval.Result{{ID: "three", IndexIdentity: "shared", Embedding: []float64{1, 0, 0}}})
	})

	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	var pollErr error
	for {
		var waiting int
		pollErr = blocker.QueryRow(ctx, `SELECT COUNT(*) FROM pg_locks WHERE locktype = 'relation' AND relation = $1::regclass AND mode = 'ShareRowExclusiveLock' AND NOT granted`, table).Scan(&waiting)
		if pollErr != nil || waiting == 2 {
			break
		}
		select {
		case <-ctx.Done():
			pollErr = ctx.Err()
		case <-ticker.C:
			continue
		}
		break
	}

	releaseErr := blocker.Commit(ctx)
	if releaseErr != nil {
		_ = blocker.Rollback(context.Background())
	}
	workerErrors := []error{<-results, <-results}
	workers.Wait()
	cancel()
	if pollErr != nil {
		t.Fatalf("wait for blocked upserts: %v", pollErr)
	}
	if releaseErr != nil {
		t.Fatalf("release blocker: %v", releaseErr)
	}

	var successes, mismatches int
	for _, workerErr := range workerErrors {
		switch {
		case workerErr == nil:
			successes++
		case errors.Is(workerErr, indexcontract.ErrDimensionMismatch):
			mismatches++
		default:
			t.Fatalf("concurrent Upsert error = %v", workerErr)
		}
	}
	if successes != 1 || mismatches != 1 {
		t.Fatalf("concurrent Upserts: successes=%d dimension_mismatches=%d errors=%v", successes, mismatches, workerErrors)
	}

	var rows, dimensions int
	queryCtx, queryCancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer queryCancel()
	if err := pool.QueryRow(queryCtx, fmt.Sprintf(`SELECT COUNT(*), COUNT(DISTINCT vector_dims(embedding)) FROM %s WHERE embedding IS NOT NULL`, idx.quoted)).Scan(&rows, &dimensions); err != nil {
		t.Fatalf("inspect stored dimensions: %v", err)
	}
	if rows != 1 || dimensions != 1 {
		t.Fatalf("stored rows=%d distinct_dimensions=%d, want 1 and 1", rows, dimensions)
	}
}

func TestStateMigrationPersistsAfterDeleteAndResetClears(t *testing.T) {
	ctx := t.Context()
	pool := openPostgresTestPool(t, ctx)
	table := fmt.Sprintf("reliquary_state_test_%d_%d", os.Getpid(), postgresTableSequence.Add(1))
	idx, err := New(pool, Config{Table: table})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, idx.quoted))
		_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, idx.stateQuoted))
	})
	for range 2 {
		if err := idx.Migrate(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if err := idx.Upsert(ctx, []*retrieval.Result{{ID: "a", DocumentID: "doc", IndexIdentity: "model-a", Embedding: []float64{1, 0}}}); err != nil {
		t.Fatal(err)
	}
	if err := idx.DeleteDocument(ctx, "doc"); err != nil {
		t.Fatal(err)
	}
	reopened, err := New(pool, Config{Table: table})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reopened.Search(ctx, indexcontract.IndexQuery{Identity: "model-b", Vector: []float32{1, 0}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
		t.Fatalf("empty reopened search mismatch = %v", err)
	}
	if err := reopened.Upsert(ctx, []*retrieval.Result{{ID: "b", IndexIdentity: "model-a", Embedding: []float64{1, 0, 0}}}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
		t.Fatalf("empty reopened upsert dimension mismatch = %v", err)
	}
	if err := reopened.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	if err := reopened.Upsert(ctx, []*retrieval.Result{{ID: "b", IndexIdentity: "model-b", Embedding: []float64{1, 0, 0}}}); err != nil {
		t.Fatalf("upsert after reset: %v", err)
	}
}

func TestMigrateBackfillsLegacyStateAndRejectsConflictingRows(t *testing.T) {
	ctx := t.Context()
	pool := openPostgresTestPool(t, ctx)
	if _, err := pool.Exec(ctx, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		t.Fatalf("enable pgvector: %v", err)
	}
	table := fmt.Sprintf("reliquary_legacy_test_%d_%d", os.Getpid(), postgresTableSequence.Add(1))
	idx, err := New(pool, Config{Table: table})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, idx.quoted))
		_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, idx.stateQuoted))
	})
	if _, err := pool.Exec(ctx, fmt.Sprintf(`CREATE TABLE %s (id text PRIMARY KEY, document_id text NOT NULL DEFAULT '', filename text NOT NULL DEFAULT '', content text NOT NULL DEFAULT '', metadata jsonb NOT NULL DEFAULT '{}'::jsonb, embedding vector)`, idx.quoted)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, fmt.Sprintf(`INSERT INTO %s (id, embedding) VALUES ('legacy', '[1,0]'::vector)`, idx.quoted)); err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); err != nil {
		t.Fatalf("Migrate legacy table: %v", err)
	}
	if _, err := idx.Search(ctx, indexcontract.IndexQuery{Identity: "new", Vector: []float32{1, 0}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
		t.Fatalf("backfilled identity mismatch = %v", err)
	}
	if _, err := pool.Exec(ctx, fmt.Sprintf(`UPDATE %s SET index_identity = 'drift' WHERE id = 'legacy'`, idx.quoted)); err != nil {
		t.Fatal(err)
	}
	if err := idx.Migrate(ctx); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
		t.Fatalf("Migrate conflicting rows = %v", err)
	}
	var identity string
	if err := pool.QueryRow(ctx, fmt.Sprintf(`SELECT index_identity FROM %s WHERE id = 'legacy'`, idx.quoted)).Scan(&identity); err != nil || identity != "drift" {
		t.Fatalf("legacy identity after failed migration = %q, err = %v", identity, err)
	}
}

func openPostgresTestPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("RELIQUARY_POSTGRES_TEST_URL")
	if url == "" {
		url = os.Getenv("DATABASE_URL")
	}
	if url == "" {
		t.Skip("PostgreSQL Index contract unavailable: set RELIQUARY_POSTGRES_TEST_URL or DATABASE_URL")
	}
	config, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatalf("parse configured PostgreSQL test URL: %v", err)
	}
	config.MaxConns = max(config.MaxConns, 3)
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatalf("connect to configured PostgreSQL test database: %v", err)
	}
	t.Cleanup(pool.Close)
	if err := pool.Ping(ctx); err != nil {
		t.Fatalf("ping configured PostgreSQL test database: %v", err)
	}
	return pool
}

func TestBoundedIndexNameAndValidateIdentifier(t *testing.T) {
	t.Parallel()

	// Short name
	name := boundedIndexName("my_table", "doc")
	if name != "idx_my_table_doc" {
		t.Fatalf("boundedIndexName = %q, want idx_my_table_doc", name)
	}

	// Long table name that causes >63 char name
	longTable := strings.Repeat("a", 60)
	longName := boundedIndexName(longTable, "suffix_very_long")
	if len(longName) != 63 {
		t.Fatalf("len(boundedIndexName) = %d, want 63", len(longName))
	}
	otherLongName := boundedIndexName(longTable, "suffix_very_different")
	if longName == otherLongName {
		t.Fatalf("boundedIndexName collision = %q", longName)
	}
	if longName != boundedIndexName(longTable, "suffix_very_long") {
		t.Fatal("boundedIndexName is not deterministic")
	}
	stateName := boundedStateTableName(longTable)
	if len(stateName) != 63 || stateName == longTable[:57]+"_state" {
		t.Fatalf("boundedStateTableName = %q (len %d)", stateName, len(stateName))
	}
	if stateName != boundedStateTableName(longTable) {
		t.Fatal("boundedStateTableName is not deterministic")
	}

	// validateIdentifier edge cases
	invalidIdentifiers := []string{
		"",
		strings.Repeat("a", 64),
		"  trimmed  ",
		"1leading_digit",
		"invalid-char!",
		"non_ascii_\u00e9",
	}
	for _, id := range invalidIdentifiers {
		if err := validateIdentifier(id); err == nil {
			t.Fatalf("validateIdentifier(%q) expected error, got nil", id)
		}
	}

	validIdentifiers := []string{
		"valid_name",
		"Table123",
		"a",
	}
	for _, id := range validIdentifiers {
		if err := validateIdentifier(id); err != nil {
			t.Fatalf("validateIdentifier(%q) unexpected error: %v", id, err)
		}
	}
}

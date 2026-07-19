package postgres

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/index/indextest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

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

func TestSearchQueryRejectsStructuredMetadataFilter(t *testing.T) {
	t.Parallel()

	idx := &Index{quoted: `"items"`}
	_, _, err := idx.searchQuery(indexcontract.IndexQuery{Filter: map[string]any{"tags": []string{"go"}}})
	if err == nil {
		t.Fatal("structured filter succeeded")
	}
}

func TestIndexContract(t *testing.T) {
	url := os.Getenv("RELIQUARY_POSTGRES_TEST_URL")
	if url == "" {
		url = os.Getenv("DATABASE_URL")
	}
	if url == "" {
		t.Skip("PostgreSQL Index contract unavailable: set RELIQUARY_POSTGRES_TEST_URL or DATABASE_URL")
	}

	ctx := context.Background()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Skipf("PostgreSQL Index contract unavailable: connect: %v", err)
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		t.Skipf("PostgreSQL Index contract unavailable: ping: %v", err)
	}

	var sequence atomic.Uint32
	tables := make([]string, 0, 8)
	t.Cleanup(func() {
		for _, table := range tables {
			_, _ = pool.Exec(context.Background(), fmt.Sprintf(`DROP TABLE IF EXISTS %s`, (pgx.Identifier{table}).Sanitize()))
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

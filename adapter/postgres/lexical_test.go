package postgres

import (
	"strings"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestLexicalConfigValidation(t *testing.T) {
	t.Parallel()
	pool := &pgxpool.Pool{}
	for _, cfg := range []Config{
		{LexicalCandidateLimit: -1},
		{LexicalCandidateLimit: 1},
		{LexicalCandidateLimit: 1, TextSearchConfig: "pg_catalog.english"},
		{LexicalCandidateLimit: 1, TextSearchConfig: "english; DROP TABLE items"},
	} {
		if _, err := New(pool, cfg); err == nil {
			t.Errorf("New(%+v) succeeded", cfg)
		}
	}
	for _, cfg := range []Config{{}, {TextSearchConfig: "english"}, {LexicalCandidateLimit: 2, TextSearchConfig: "simple"}} {
		if _, err := New(pool, cfg); err != nil {
			t.Errorf("New(%+v): %v", cfg, err)
		}
	}
}

func TestLexicalCandidateQueryShape(t *testing.T) {
	t.Parallel()
	idx, err := New(&pgxpool.Pool{}, Config{LexicalCandidateLimit: 4, TextSearchConfig: "english"})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, text string
		vector     []float32
		limit      int
		wantCap    int
		lexical    bool
	}{
		{name: "text", text: "quartz beacon", limit: 7, wantCap: 7, lexical: true},
		{name: "configured cap", text: "quartz beacon", limit: 1, wantCap: 4, lexical: true},
		{name: "zero query limit", text: "quartz beacon", wantCap: 4, lexical: true},
		{name: "hybrid", text: "quartz beacon", vector: []float32{1, 0}, limit: 1},
		{name: "vector", vector: []float32{1, 0}, limit: 1},
		{name: "empty text", limit: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			statement, args, err := idx.searchQuery(indexcontract.IndexQuery{
				Text: tc.text, Vector: tc.vector, Limit: tc.limit,
				Filter: map[string]any{"document_id": "doc", "tenant": "allowed"},
			})
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(statement, "plainto_tsquery") != tc.lexical {
				t.Fatalf("unexpected lexical SQL: %s", statement)
			}
			if !tc.lexical {
				if len(tc.vector) > 0 && !strings.Contains(statement, `embedding <=> $3 NULLS LAST, id ASC LIMIT $4`) {
					t.Fatalf("vector order changed: %s", statement)
				}
				return
			}
			for _, want := range []string{
				`document_id = $1`, `metadata @> $2::jsonb`,
				`to_tsvector($3::regconfig, filename || ' ' || content) @@ plainto_tsquery($3::regconfig, $4)`,
				`ORDER BY ts_rank_cd(to_tsvector($3::regconfig, filename || ' ' || content), plainto_tsquery($3::regconfig, $4)) DESC, id ASC LIMIT $5`,
			} {
				if !strings.Contains(statement, want) {
					t.Errorf("query %q missing %q", statement, want)
				}
			}
			if len(args) != 5 || args[0] != "doc" || args[2] != "english" || args[3] != tc.text || args[4] != tc.wantCap {
				t.Fatalf("args = %#v", args)
			}
			if strings.Contains(statement, "english") || strings.Contains(statement, "quartz beacon") {
				t.Fatalf("caller values interpolated into SQL: %s", statement)
			}
		})
	}
}

func TestLexicalDisabledPreservesExhaustiveTextQuery(t *testing.T) {
	t.Parallel()
	idx, err := New(&pgxpool.Pool{}, Config{})
	if err != nil {
		t.Fatal(err)
	}
	statement, args, err := idx.searchQuery(indexcontract.IndexQuery{Text: "quartz beacon", Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(statement, " LIMIT ") || strings.Contains(statement, "tsquery") || len(args) != 0 {
		t.Fatalf("default text search changed: %q %#v", statement, args)
	}
}

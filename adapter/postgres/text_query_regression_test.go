package postgres

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
	"github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"
)

func TestTextSearchQueryRanksBeforeLimit(t *testing.T) {
	t.Parallel()
	idx := &Index{quoted: `"items"`}
	statement, args, err := idx.searchQuery(indexcontract.IndexQuery{
		Text: "quartz beacon", Limit: 1, Filter: map[string]any{"document_id": "doc", "tenant": "allowed"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(statement, " LIMIT ") {
		t.Fatalf("text query truncated before ranking: %s", statement)
	}
	for _, want := range []string{`document_id = $1`, `metadata @> $2::jsonb`, `ORDER BY id ASC`} {
		if !strings.Contains(statement, want) {
			t.Fatalf("query %q missing %q", statement, want)
		}
	}
	if len(args) != 2 || args[0] != "doc" {
		t.Fatalf("filter args = %#v", args)
	}
	var filter map[string]any
	if err := json.Unmarshal(args[1].([]byte), &filter); err != nil || filter["tenant"] != "allowed" {
		t.Fatalf("metadata filter = %v, error = %v", filter, err)
	}
}

func TestSearchQueryVectorBoundsAndZeroLimit(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, text string
		vector     []float32
		limit      int
		bounded    bool
	}{
		{"vector", "", []float32{1, 0}, 1, true},
		{"hybrid", "quartz beacon", []float32{1, 0}, 1, true},
		{"unbounded vector", "", []float32{1, 0}, 0, false},
		{"unbounded text", "quartz beacon", nil, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			statement, _, err := (&Index{quoted: `"items"`}).searchQuery(indexcontract.IndexQuery{Text: tc.text, Vector: tc.vector, Limit: tc.limit})
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(statement, " LIMIT ") != tc.bounded {
				t.Fatalf("query bounds: %s", statement)
			}
		})
	}
}

type fixtureRows struct {
	pgx.Rows
	items  []*retrieval.Result
	next   int
	closed bool
}

func (r *fixtureRows) Next() bool {
	if r.next >= len(r.items) {
		return false
	}
	r.next++
	return true
}
func (r *fixtureRows) Close()     { r.closed = true }
func (r *fixtureRows) Err() error { return nil }
func (r *fixtureRows) Scan(dest ...any) error {
	item := r.items[r.next-1]
	*dest[0].(*string), *dest[1].(*string), *dest[2].(*string), *dest[3].(*string) = item.ID, item.DocumentID, item.Filename, item.Content
	metadata, err := json.Marshal(item.Metadata)
	if err != nil {
		return err
	}
	*dest[4].(*[]byte) = metadata
	*dest[5].(**pgvector.Vector) = nil
	*dest[6].(*string) = item.IndexIdentity
	return nil
}

func TestTextSearchTransportReturnsLateRelevanceWinner(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	t.Cleanup(cancel)
	rows := &fixtureRows{items: []*retrieval.Result{
		{ID: "a", Content: "unrelated material", Metadata: map[string]any{"tenant": "allowed"}},
		{ID: "z", Content: "quartz beacon", Metadata: map[string]any{"tenant": "allowed"}},
	}}
	committed := false
	tx := &fixtureTx{
		queryRow: func(c context.Context, _ string, _ ...any) pgx.Row {
			if c != ctx {
				t.Error("state query lost operation context")
			}
			return fixtureRow(func(...any) error { return pgx.ErrNoRows })
		},
		query: func(c context.Context, sql string, _ ...any) (pgx.Rows, error) {
			if c != ctx {
				t.Error("query lost operation context")
			}
			if !strings.Contains(sql, "metadata @>") {
				t.Error("SQL filter missing")
			}
			// Model the old premature truncation without running a database engine.
			if strings.Contains(sql, " LIMIT ") {
				rows.items = rows.items[:1]
			}
			return rows, nil
		},
		commit: func(c context.Context) error {
			if c != ctx {
				t.Error("commit lost operation context")
			}
			committed = true
			return nil
		},
		rollback: func(context.Context) error { return pgx.ErrTxClosed },
	}
	idx := &Index{pool: &fixturePool{tx: tx}, quoted: `"items"`, stateQuoted: `"state"`}
	got, err := idx.Search(ctx, indexcontract.IndexQuery{Text: "quartz beacon", Limit: 1, Filter: map[string]any{"tenant": "allowed"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "z" {
		t.Fatalf("text winner = %v, want z", got)
	}
	if !rows.closed || !committed {
		t.Fatalf("rows closed = %v, committed = %v", rows.closed, committed)
	}
}

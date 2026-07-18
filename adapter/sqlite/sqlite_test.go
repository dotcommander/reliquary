package sqlite

import (
	"context"
	"database/sql"
	"errors"
	"slices"
	"strings"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/index/indextest"
	"github.com/dotcommander/reliquary/retrieval"
	_ "modernc.org/sqlite"
)

func TestIndexContract(t *testing.T) {
	indextest.Run(t, func() indexcontract.Index {
		db, err := sql.Open("sqlite", ":memory:")
		if err != nil {
			t.Fatal(err)
		}
		db.SetMaxOpenConns(1)
		idx, err := New(db, Config{})
		if err != nil {
			t.Fatal(err)
		}
		if err := idx.Migrate(context.Background()); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { db.Close() })
		return idx
	})
}

func TestNewValidatesWithoutDatabaseIO(t *testing.T) {
	db := &sql.DB{}
	if _, err := New(db, Config{Table: "bad-name"}); err == nil {
		t.Fatal("expected invalid table")
	}
	if _, err := New(nil, Config{}); err == nil {
		t.Fatal("expected nil db")
	}
	if _, err := New(db, Config{Table: "valid_name"}); err != nil {
		t.Fatalf("validation-only New: %v", err)
	}
}

func TestMigrateIsIdempotentAndFTSBoundsCandidates(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{CandidateLimit: 2})
	for range 2 {
		if err := idx.Migrate(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	items := []*retrieval.Result{{ID: "a", Content: "needle exact", Embedding: []float64{0, 1}}, {ID: "b", Content: "unrelated", Embedding: []float64{1, 0}}, {ID: "c", Content: "needle also", Embedding: []float64{1, 0}}}
	if err := idx.Upsert(context.Background(), items); err != nil {
		t.Fatal(err)
	}
	got, err := idx.Search(context.Background(), indexcontract.IndexQuery{Text: "needle", Vector: []float32{1, 0}, Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "c" {
		t.Fatalf("FTS search = %#v", got)
	}
}

func TestSearchTreatsTextAsPlainTextInsteadOfFTSSyntax(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	items := []*retrieval.Result{
		{ID: "cpp", Content: `A C++ guide with operators (and quotes).`},
		{ID: "other", Content: "A plain C guide."},
	}
	if err := idx.Upsert(context.Background(), items); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{"C++", `operators (and`, `quotes).`} {
		got, err := idx.Search(context.Background(), indexcontract.IndexQuery{Text: query, Limit: 5})
		if err != nil {
			t.Fatalf("Search(%q): %v", query, err)
		}
		if !slices.ContainsFunc(got, func(result *retrieval.Result) bool { return result.ID == "cpp" }) {
			t.Fatalf("Search(%q) = %#v", query, got)
		}
	}
}

func TestMigrateAddsIdentityToExistingTableAndPersistsIt(t *testing.T) {
	db := openTestDB(t)
	if _, err := db.Exec(`CREATE TABLE reliquary_index (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO reliquary_index VALUES ('legacy', '', '', '', '{}', '[1,0]')`); err != nil {
		t.Fatal(err)
	}
	idx, _ := New(db, Config{})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	item := &retrieval.Result{ID: "a", IndexIdentity: "model-a|chunks-v1", Embedding: []float64{1, 0}}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{item}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
		t.Fatalf("legacy identity mismatch = %v", err)
	}
	if err := idx.Reset(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{item}); err != nil {
		t.Fatal(err)
	}
	reopened, _ := New(db, Config{})
	if _, err := reopened.Search(context.Background(), indexcontract.IndexQuery{Identity: "model-b|chunks-v1", Vector: []float32{1, 0}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
		t.Fatalf("reopened identity mismatch error = %v", err)
	}
	got, err := reopened.Search(context.Background(), indexcontract.IndexQuery{Identity: item.IndexIdentity, Vector: []float32{1, 0}})
	if err != nil || len(got) != 1 || got[0].IndexIdentity != item.IndexIdentity {
		t.Fatalf("reopened search = %#v, %v", got, err)
	}
}

func TestVectorOnlyBoundedFallbackUsesStableIDs(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{CandidateLimit: 2})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "a", Embedding: []float64{0, 1}}, {ID: "b", Embedding: []float64{0, 1}}, {ID: "z", Embedding: []float64{1, 0}}}); err != nil {
		t.Fatal(err)
	}
	got, err := idx.Search(context.Background(), indexcontract.IndexQuery{Vector: []float32{1, 0}, Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("bounded stable-ID candidates = %#v", got)
	}
}

func TestScalarFilterAndTransactionalRollback(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	bad := make(chan int)
	err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "kept", Metadata: map[string]any{"tenant": "one", "active": true}}, {ID: "bad", Metadata: map[string]any{"bad": bad}}})
	if err == nil {
		t.Fatal("expected JSON encoding failure")
	}
	var count int
	if err := db.QueryRow("SELECT count(*) FROM reliquary_index").Scan(&count); err != nil || count != 0 {
		t.Fatalf("count=%d err=%v", count, err)
	}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "kept", Metadata: map[string]any{"tenant": "one", "active": true}}}); err != nil {
		t.Fatal(err)
	}
	got, err := idx.Search(context.Background(), indexcontract.IndexQuery{Filter: map[string]any{"tenant": "one", "active": true}})
	if err != nil || len(got) != 1 {
		t.Fatalf("got=%#v err=%v", got, err)
	}
	if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Filter: map[string]any{"tenant": []string{"one"}}}); err == nil || !strings.Contains(err.Error(), "must be scalar") {
		t.Fatalf("error=%v", err)
	}
}

func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	return db
}

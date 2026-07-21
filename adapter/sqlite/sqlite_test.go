package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

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

func TestMigrateRebuildsFTSFromLegacyBaseTable(t *testing.T) {
	db := openTestDB(t)
	if _, err := db.Exec(`CREATE TABLE reliquary_index (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`CREATE VIRTUAL TABLE reliquary_index_fts USING fts5(id UNINDEXED, content, filename)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO reliquary_index VALUES
		('current', 'doc-current', 'current.md', 'current searchable text', '{}', '[]'),
		('missing', 'doc-missing', 'missing.md', 'missing searchable text', '{}', '[]')`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO reliquary_index_fts VALUES
		('current', 'stale searchable text', 'old.md'),
		('orphan', 'orphan searchable text', 'orphan.md')`); err != nil {
		t.Fatal(err)
	}

	idx, err := New(db, Config{})
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := idx.Migrate(context.Background()); err != nil {
			t.Fatal(err)
		}
	}

	for query, wantID := range map[string]string{
		"current": "current",
		"missing": "missing",
	} {
		got, err := idx.Search(context.Background(), indexcontract.IndexQuery{Text: query})
		if err != nil {
			t.Fatalf("Search(%q): %v", query, err)
		}
		if len(got) != 1 || got[0].ID != wantID {
			t.Fatalf("Search(%q) = %#v, want %q", query, got, wantID)
		}
	}
	for _, query := range []string{"stale", "orphan"} {
		got, err := idx.Search(context.Background(), indexcontract.IndexQuery{Text: query})
		if err != nil {
			t.Fatalf("Search(%q): %v", query, err)
		}
		if len(got) != 0 {
			t.Fatalf("Search(%q) = %#v, want no results", query, got)
		}
	}

	var missing, orphan int
	if err := db.QueryRow(`
		SELECT
			(SELECT COUNT(*) FROM reliquary_index AS r LEFT JOIN reliquary_index_fts AS f ON f.id = r.id WHERE f.id IS NULL),
			(SELECT COUNT(*) FROM reliquary_index_fts AS f LEFT JOIN reliquary_index AS r ON r.id = f.id WHERE r.id IS NULL)
	`).Scan(&missing, &orphan); err != nil {
		t.Fatal(err)
	}
	if missing != 0 || orphan != 0 {
		t.Fatalf("FTS parity after migration: missing=%d orphan=%d", missing, orphan)
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

func TestStatePersistsAfterDeleteAndResetClearsIt(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{})
	for range 2 {
		if err := idx.Migrate(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	item := &retrieval.Result{ID: "a", DocumentID: "doc", IndexIdentity: "model-a", Embedding: []float64{1, 0}}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{item}); err != nil {
		t.Fatal(err)
	}
	if err := idx.DeleteDocument(context.Background(), "doc"); err != nil {
		t.Fatal(err)
	}
	reopened, _ := New(db, Config{})
	if _, err := reopened.Search(context.Background(), indexcontract.IndexQuery{Identity: "model-b", Vector: []float32{1, 0}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
		t.Fatalf("empty reopened search mismatch = %v", err)
	}
	if err := reopened.Upsert(context.Background(), []*retrieval.Result{{ID: "b", IndexIdentity: "model-a", Embedding: []float64{1, 0, 0}}}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
		t.Fatalf("empty reopened upsert dimension mismatch = %v", err)
	}
	if err := reopened.Reset(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := reopened.Upsert(context.Background(), []*retrieval.Result{{ID: "b", IndexIdentity: "model-b", Embedding: []float64{1, 0, 0}}}); err != nil {
		t.Fatalf("upsert after reset: %v", err)
	}
}

func TestMigrateRejectsConflictingLegacyDimensionsWithoutChangingRows(t *testing.T) {
	db := openTestDB(t)
	if _, err := db.Exec(`CREATE TABLE reliquary_index (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL, index_identity TEXT NOT NULL DEFAULT '')`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO reliquary_index VALUES ('a', '', '', '', '{}', '[1,0]', 'legacy'), ('b', '', '', '', '{}', '[1,0,0]', 'legacy')`); err != nil {
		t.Fatal(err)
	}
	idx, _ := New(db, Config{})
	if err := idx.Migrate(context.Background()); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
		t.Fatalf("Migrate error = %v", err)
	}
	var count int
	if err := db.QueryRow(`SELECT COUNT(*) FROM reliquary_index`).Scan(&count); err != nil || count != 2 {
		t.Fatalf("legacy row count = %d, err = %v", count, err)
	}
	var stateTableCount int
	if err := db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='reliquary_index_state'`).Scan(&stateTableCount); err != nil || stateTableCount != 0 {
		t.Fatalf("state table count after rolled-back migration = %d, err = %v", stateTableCount, err)
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

func TestZeroLimitUsesBoundedStableIDCandidatePool(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{CandidateLimit: 2})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{
		{ID: "a", Embedding: []float64{0, 1}},
		{ID: "b", Embedding: []float64{0, 1}},
		{ID: "z", Embedding: []float64{1, 0}},
	}); err != nil {
		t.Fatal(err)
	}

	got, err := idx.Search(context.Background(), indexcontract.IndexQuery{Vector: []float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	if ids := resultIDsForTest(got); ids != "a,b" {
		t.Fatalf("zero-limit bounded candidates = %s, want a,b", ids)
	}
}

func resultIDsForTest(results []*retrieval.Result) string {
	ids := make([]string, len(results))
	for n, result := range results {
		ids[n] = result.ID
	}
	return strings.Join(ids, ",")
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

func TestUpsertPanicRollsBackAndReleasesConnection(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}

	panicValue := errors.New("metadata marshal panic")
	var recovered any
	func() {
		defer func() { recovered = recover() }()
		_ = idx.Upsert(context.Background(), []*retrieval.Result{
			{ID: "first", Content: "first searchable row"},
			{ID: "second", Metadata: map[string]any{"panic": panicMarshaler{value: panicValue}}},
		})
	}()
	if recovered != panicValue {
		t.Fatalf("recovered panic = %#v, want original %#v", recovered, panicValue)
	}
	if inUse := db.Stats().InUse; inUse != 0 {
		t.Fatalf("connections in use after panic = %d, want 0", inUse)
	}

	var relationalCount, ftsCount int
	if err := db.QueryRow(`SELECT (SELECT COUNT(*) FROM reliquary_index), (SELECT COUNT(*) FROM reliquary_index_fts)`).Scan(&relationalCount, &ftsCount); err != nil {
		t.Fatal(err)
	}
	if relationalCount != 0 || ftsCount != 0 {
		t.Fatalf("rows after panic: relational=%d FTS=%d, want both 0", relationalCount, ftsCount)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := idx.Upsert(ctx, []*retrieval.Result{{ID: "valid", Content: "reusable connection"}}); err != nil {
		t.Fatalf("upsert after panic: %v", err)
	}
	got, err := idx.Search(ctx, indexcontract.IndexQuery{Text: "reusable", Limit: 1})
	if err != nil {
		t.Fatalf("search after panic: %v", err)
	}
	if len(got) != 1 || got[0].ID != "valid" {
		t.Fatalf("search after panic = %#v", got)
	}
}

func TestMetadataFilterPreselectionPreservesJSONTypes(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{CandidateLimit: 1})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{
		{ID: "a-number", Metadata: map[string]any{"value": 1}},
		{ID: "b-boolean", Metadata: map[string]any{"value": true}},
	}); err != nil {
		t.Fatal(err)
	}

	got, err := idx.Search(context.Background(), indexcontract.IndexQuery{Filter: map[string]any{"value": true}, Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "b-boolean" {
		t.Fatalf("boolean-filtered candidates = %#v", got)
	}
}

func TestNumericFilterPreselectionDoesNotSpendLimitOnNearValue(t *testing.T) {
	db := openTestDB(t)
	idx, _ := New(db, Config{CandidateLimit: 1})
	if err := idx.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := idx.Upsert(context.Background(), []*retrieval.Result{
		{ID: "a-near", Metadata: map[string]any{"value": uint64(9007199254740992)}},
		{ID: "b-exact", Metadata: map[string]any{"value": uint64(9007199254740993)}},
	}); err != nil {
		t.Fatal(err)
	}

	got, err := idx.Search(context.Background(), indexcontract.IndexQuery{
		Filter: map[string]any{"value": uint64(9007199254740993)},
		Limit:  1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "b-exact" {
		t.Fatalf("numeric-filtered candidates = %#v", got)
	}
}

type panicMarshaler struct {
	value any
}

func (m panicMarshaler) MarshalJSON() ([]byte, error) {
	panic(m.value)
}

var _ json.Marshaler = panicMarshaler{}

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

// Package sqlite provides a SQLite-backed Reliquary candidate index.
package sqlite

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/internal/indexutil"
	"github.com/dotcommander/reliquary/internal/sqltx"
	"github.com/dotcommander/reliquary/retrieval"
)

const defaultTable = "reliquary_index"
const defaultCandidates = 100

var identifierPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// Config controls schema ownership and bounded candidate retrieval.
type Config struct {
	Table          string
	CandidateLimit int
}

// Index uses caller-owned SQLite connectivity. New performs no database I/O.
type Index struct {
	db             *sql.DB
	table          string
	ftsTable       string
	stateTable     string
	candidateLimit int
}

func New(db *sql.DB, cfg Config) (*Index, error) {
	if db == nil {
		return nil, fmt.Errorf("sqlite index: nil db")
	}
	if cfg.Table == "" {
		cfg.Table = defaultTable
	}
	if !identifierPattern.MatchString(cfg.Table) {
		return nil, fmt.Errorf("sqlite index: invalid table %q", cfg.Table)
	}
	if cfg.CandidateLimit < 0 {
		return nil, fmt.Errorf("sqlite index: candidate limit must not be negative")
	}
	if cfg.CandidateLimit == 0 {
		cfg.CandidateLimit = defaultCandidates
	}
	return &Index{db: db, table: cfg.Table, ftsTable: cfg.Table + "_fts", stateTable: cfg.Table + "_state", candidateLimit: cfg.CandidateLimit}, nil
}

// Migrate creates only the retrieval-owned schema and is safe to repeat.
func (i *Index) Migrate(ctx context.Context) error {
	return withTx(ctx, i.db, func(tx *sql.Tx) error {
		statements := []string{
			fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL, index_identity TEXT NOT NULL DEFAULT '')`, i.table),
			fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_document_id ON %s(document_id)`, i.table, i.table),
			fmt.Sprintf(`CREATE VIRTUAL TABLE IF NOT EXISTS %s USING fts5(id UNINDEXED, content, filename)`, i.ftsTable),
			fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), index_identity TEXT, embedding_dimension INTEGER CHECK (embedding_dimension IS NULL OR embedding_dimension > 0))`, i.stateTable),
		}
		for _, statement := range statements {
			if _, err := tx.ExecContext(ctx, statement); err != nil {
				return fmt.Errorf("migrate sqlite index: %w", err)
			}
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`ALTER TABLE %s ADD COLUMN index_identity TEXT NOT NULL DEFAULT ''`, i.table)); err != nil && !strings.Contains(err.Error(), "duplicate column name") {
			return fmt.Errorf("migrate sqlite index identity: %w", err)
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s`, i.ftsTable)); err != nil {
			return fmt.Errorf("rebuild sqlite FTS index: %w", err)
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (id, content, filename) SELECT id, content, filename FROM %s`, i.ftsTable, i.table)); err != nil {
			return fmt.Errorf("repopulate sqlite FTS index: %w", err)
		}
		return i.backfillState(ctx, tx)
	})
}

func (i *Index) Upsert(ctx context.Context, items []*retrieval.Result) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return withTx(ctx, i.db, func(tx *sql.Tx) error {
		space, err := i.readState(ctx, tx)
		if err != nil {
			return err
		}
		for _, item := range items {
			if err := ctx.Err(); err != nil {
				return err
			}
			if item == nil {
				continue
			}
			if item.ID == "" {
				return fmt.Errorf("sqlite index: empty item ID")
			}
		}
		space, err = space.ValidateResults(items)
		if err != nil {
			return err
		}
		if err := i.writeState(ctx, tx, space); err != nil {
			return err
		}
		for _, item := range items {
			if item == nil {
				continue
			}
			if err := i.upsertItem(ctx, tx, item); err != nil {
				return err
			}
		}
		return nil
	})
}

// ReplaceDocuments atomically replaces complete document revisions in both
// the base and FTS tables. Empty result sets delete their document.
func (i *Index) ReplaceDocuments(ctx context.Context, replacements []indexcontract.DocumentReplacement) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := indexutil.ValidateReplacements(replacements); err != nil {
		return err
	}
	return withTx(ctx, i.db, func(tx *sql.Tx) error {
		space, err := i.readState(ctx, tx)
		if err != nil {
			return err
		}
		var replacementItems []*retrieval.Result
		for _, replacement := range replacements {
			replacementItems = append(replacementItems, replacement.Results...)
		}
		space, err = space.ValidateResults(replacementItems)
		if err != nil {
			return err
		}
		for _, replacement := range replacements {
			if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE id IN (SELECT id FROM %s WHERE document_id = ?)`, i.ftsTable, i.table), replacement.DocumentID); err != nil {
				return fmt.Errorf("sqlite index: delete replacement FTS document %q: %w", replacement.DocumentID, err)
			}
			if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE document_id = ?`, i.table), replacement.DocumentID); err != nil {
				return fmt.Errorf("sqlite index: delete replacement document %q: %w", replacement.DocumentID, err)
			}
		}
		for _, replacement := range replacements {
			for _, item := range replacement.Results {
				if item == nil {
					continue
				}
				var retainedDocumentID string
				err := tx.QueryRowContext(ctx, fmt.Sprintf(`SELECT document_id FROM %s WHERE id = ?`, i.table), item.ID).Scan(&retainedDocumentID)
				switch {
				case err == nil:
					return fmt.Errorf("%w: %q belongs to retained document %q", indexcontract.ErrResultIDConflict, item.ID, retainedDocumentID)
				case errors.Is(err, sql.ErrNoRows):
				case err != nil:
					return fmt.Errorf("sqlite index: inspect replacement result %q: %w", item.ID, err)
				}
			}
		}
		if err := i.writeState(ctx, tx, space); err != nil {
			return err
		}
		for _, replacement := range replacements {
			for _, item := range replacement.Results {
				if err := ctx.Err(); err != nil {
					return err
				}
				if item == nil {
					continue
				}
				if err := i.upsertItem(ctx, tx, item); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

func (i *Index) upsertItem(ctx context.Context, tx *sql.Tx, item *retrieval.Result) error {
	metadata, err := json.Marshal(item.Metadata)
	if err != nil {
		return fmt.Errorf("sqlite index: encode metadata for %q: %w", item.ID, err)
	}
	embedding, err := json.Marshal(item.Embedding)
	if err != nil {
		return fmt.Errorf("sqlite index: encode embedding for %q: %w", item.ID, err)
	}
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (id, document_id, filename, content, metadata, embedding, index_identity) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET document_id=excluded.document_id, filename=excluded.filename, content=excluded.content, metadata=excluded.metadata, embedding=excluded.embedding, index_identity=excluded.index_identity`, i.table), item.ID, item.DocumentID, item.Filename, item.Content, string(metadata), string(embedding), item.IndexIdentity); err != nil {
		return fmt.Errorf("sqlite index: upsert %q: %w", item.ID, err)
	}
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE id = ?`, i.ftsTable), item.ID); err != nil {
		return fmt.Errorf("sqlite index: delete prior FTS row %q: %w", item.ID, err)
	}
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (id, content, filename) VALUES (?, ?, ?)`, i.ftsTable), item.ID, item.Content, item.Filename); err != nil {
		return fmt.Errorf("sqlite index: insert FTS row %q: %w", item.ID, err)
	}
	return nil
}

func (i *Index) DeleteDocument(ctx context.Context, documentID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := indexutil.ValidateDocumentID(documentID); err != nil {
		return err
	}
	return withTx(ctx, i.db, func(tx *sql.Tx) error {
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE id IN (SELECT id FROM %s WHERE document_id = ?)`, i.ftsTable, i.table), documentID); err != nil {
			return fmt.Errorf("sqlite index: delete FTS document: %w", err)
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE document_id = ?`, i.table), documentID); err != nil {
			return fmt.Errorf("sqlite index: delete document: %w", err)
		}
		return nil
	})
}

// Reset destructively removes all indexed results and FTS entries.
func (i *Index) Reset(ctx context.Context) error {
	return withTx(ctx, i.db, func(tx *sql.Tx) error {
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s`, i.ftsTable)); err != nil {
			return fmt.Errorf("sqlite index: reset FTS: %w", err)
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s`, i.table)); err != nil {
			return fmt.Errorf("sqlite index: reset: %w", err)
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s`, i.stateTable)); err != nil {
			return fmt.Errorf("sqlite index: reset state: %w", err)
		}
		return nil
	})
}

func (i *Index) Search(ctx context.Context, query indexcontract.IndexQuery) ([]*retrieval.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := indexutil.ValidateFilter(query.Filter); err != nil {
		return nil, fmt.Errorf("sqlite index: %w", err)
	}
	var candidates []*retrieval.Result
	err := withTx(ctx, i.db, func(tx *sql.Tx) error {
		var err error
		candidates, err = i.searchSnapshot(ctx, tx, query)
		return err
	})
	return candidates, err
}

func (i *Index) searchSnapshot(ctx context.Context, tx *sql.Tx, query indexcontract.IndexQuery) ([]*retrieval.Result, error) {
	space, err := i.readState(ctx, tx)
	if err != nil {
		return nil, err
	}
	if err := space.ValidateQuery(query); err != nil {
		return nil, err
	}
	where, args, err := buildFilters(query.Filter)
	if err != nil {
		return nil, err
	}
	limit := 0
	if query.Limit == 0 {
		limit = i.candidateLimit
	} else if query.Limit > 0 {
		limit = max(i.candidateLimit, query.Limit)
	}
	from := i.table + " AS r"
	if strings.TrimSpace(query.Text) != "" {
		from += " JOIN " + i.ftsTable + " AS f ON f.id=r.id"
		if where != "" {
			where += " AND "
		} else {
			where = " WHERE "
		}
		where += i.ftsTable + " MATCH ?"
		args = append(args, plainTextFTSQuery(query.Text))
	}
	statement := fmt.Sprintf(`SELECT r.id, r.document_id, r.filename, r.content, r.metadata, r.embedding, r.index_identity FROM %s%s ORDER BY r.id`, from, where)
	rows, err := tx.QueryContext(ctx, statement, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite index: select candidates: %w", err)
	}
	defer rows.Close()
	var candidates []*retrieval.Result
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var item retrieval.Result
		var metadata, embedding string
		if err := rows.Scan(&item.ID, &item.DocumentID, &item.Filename, &item.Content, &metadata, &embedding, &item.IndexIdentity); err != nil {
			return nil, err
		}
		decoder := json.NewDecoder(bytes.NewBufferString(metadata))
		decoder.UseNumber()
		if err := decoder.Decode(&item.Metadata); err != nil {
			return nil, err
		}
		if err := json.Unmarshal([]byte(embedding), &item.Embedding); err != nil {
			return nil, err
		}
		if !indexutil.MatchesFilter(&item, query.Filter) {
			continue
		}
		candidates = append(candidates, &item)
		if limit > 0 && len(candidates) == limit {
			break
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return indexutil.Search(ctx, query, candidates)
}

// plainTextFTSQuery treats caller text as terms, not as FTS5 query syntax.
// Quoting each whitespace-delimited term preserves the default AND behavior
// while making punctuation and operators safe.
func plainTextFTSQuery(text string) string {
	terms := strings.Fields(text)
	for n, term := range terms {
		terms[n] = `"` + strings.ReplaceAll(term, `"`, `""`) + `"`
	}
	return strings.Join(terms, " ")
}

func (i *Index) readState(ctx context.Context, q queryer) (indexutil.Space, error) {
	var identity sql.NullString
	var dimension sql.NullInt64
	err := q.QueryRowContext(ctx, fmt.Sprintf(`SELECT index_identity, embedding_dimension FROM %s WHERE singleton = 1`, i.stateTable)).Scan(&identity, &dimension)
	if errors.Is(err, sql.ErrNoRows) {
		return indexutil.Space{}, nil
	}
	if err != nil {
		return indexutil.Space{}, fmt.Errorf("sqlite index: read state: %w", err)
	}
	return indexutil.Space{Identity: identity.String, IdentitySet: identity.Valid, Dimension: int(dimension.Int64)}, nil
}

func (i *Index) writeState(ctx context.Context, tx *sql.Tx, state indexutil.Space) error {
	if !state.IdentitySet && state.Dimension == 0 {
		return nil
	}
	var identity any
	if state.IdentitySet {
		identity = state.Identity
	}
	var dimension any
	if state.Dimension > 0 {
		dimension = state.Dimension
	}
	_, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (singleton, index_identity, embedding_dimension) VALUES (1, ?, ?) ON CONFLICT(singleton) DO UPDATE SET index_identity=excluded.index_identity, embedding_dimension=excluded.embedding_dimension`, i.stateTable), identity, dimension)
	if err != nil {
		return fmt.Errorf("sqlite index: write state: %w", err)
	}
	return nil
}

func (i *Index) backfillState(ctx context.Context, tx *sql.Tx) error {
	state, err := i.readState(ctx, tx)
	if err != nil {
		return err
	}
	var minimum, maximum string
	var count int
	if err := tx.QueryRowContext(ctx, fmt.Sprintf(`SELECT COALESCE(MIN(index_identity), ''), COALESCE(MAX(index_identity), ''), COUNT(*) FROM %s`, i.table)).Scan(&minimum, &maximum, &count); err != nil {
		return fmt.Errorf("sqlite index: inspect legacy identities: %w", err)
	}
	if minimum != maximum {
		return fmt.Errorf("%w: stored rows contain multiple identities", indexcontract.ErrIdentityMismatch)
	}
	legacy := indexutil.Space{}
	if count > 0 {
		legacy.Identity, legacy.IdentitySet = minimum, true
	}
	rows, err := tx.QueryContext(ctx, fmt.Sprintf(`SELECT embedding FROM %s WHERE embedding != 'null' AND embedding != '[]'`, i.table))
	if err != nil {
		return fmt.Errorf("sqlite index: inspect legacy dimensions: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return err
		}
		var vector []float64
		if err := json.Unmarshal([]byte(raw), &vector); err != nil {
			return fmt.Errorf("sqlite index: decode legacy embedding: %w", err)
		}
		if len(vector) == 0 {
			continue
		}
		if legacy.Dimension == 0 {
			legacy.Dimension = len(vector)
		} else if legacy.Dimension != len(vector) {
			return fmt.Errorf("%w: stored rows contain multiple dimensions", indexcontract.ErrDimensionMismatch)
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if state.IdentitySet && legacy.IdentitySet && state.Identity != legacy.Identity {
		return fmt.Errorf("%w: state has %q, stored rows have %q", indexcontract.ErrIdentityMismatch, state.Identity, legacy.Identity)
	}
	if state.Dimension > 0 && legacy.Dimension > 0 && state.Dimension != legacy.Dimension {
		return fmt.Errorf("%w: state has %d dimensions, stored rows have %d", indexcontract.ErrDimensionMismatch, state.Dimension, legacy.Dimension)
	}
	if !state.IdentitySet {
		state.Identity, state.IdentitySet = legacy.Identity, legacy.IdentitySet
	}
	if state.Dimension == 0 {
		state.Dimension = legacy.Dimension
	}
	return i.writeState(ctx, tx, state)
}

type queryer interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func buildFilters(filters map[string]any) (string, []any, error) {
	if err := indexutil.ValidateFilter(filters); err != nil {
		return "", nil, fmt.Errorf("sqlite index: %w", err)
	}
	keys := make([]string, 0, len(filters))
	for key := range filters {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	var clauses []string
	var args []any
	for _, key := range keys {
		value := filters[key]
		switch key {
		case "id", "document_id", "filename":
			if _, ok := value.(string); !ok {
				clauses = append(clauses, "0")
				continue
			}
			clauses = append(clauses, "r."+key+" = ?")
			args = append(args, value)
		default:
			if indexutil.IsJSONNumber(value) {
				clauses = append(clauses, `EXISTS (
					SELECT 1 FROM json_each(r.metadata) AS actual
					WHERE actual.key = ? AND actual.type IN ('integer', 'real')
				)`)
				args = append(args, key)
				continue
			}
			expected, err := json.Marshal(map[string]any{key: value})
			if err != nil {
				return "", nil, fmt.Errorf("sqlite index: encode filter %q: %w", key, err)
			}
			clauses = append(clauses, `EXISTS (
				SELECT 1
				FROM json_each(r.metadata) AS actual
				JOIN json_each(?) AS expected
				  ON actual.key = expected.key
				 AND actual.type = expected.type
				 AND actual.atom IS expected.atom
			)`)
			args = append(args, string(expected))
		}
	}
	if len(clauses) == 0 {
		return "", args, nil
	}
	return " WHERE " + strings.Join(clauses, " AND "), args, nil
}

func withTx(ctx context.Context, db *sql.DB, fn func(*sql.Tx) error) error {
	return sqltx.Run(
		func() (*sql.Tx, error) { return db.BeginTx(ctx, nil) },
		func(tx *sql.Tx) error { return tx.Rollback() },
		func(tx *sql.Tx) error { return tx.Commit() },
		fn,
	)
}

var _ indexcontract.Index = (*Index)(nil)

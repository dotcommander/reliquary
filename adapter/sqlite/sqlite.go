// Package sqlite provides a SQLite-backed Reliquary candidate index.
package sqlite

import (
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
	return &Index{db: db, table: cfg.Table, ftsTable: cfg.Table + "_fts", candidateLimit: cfg.CandidateLimit}, nil
}

// Migrate creates only the retrieval-owned schema and is safe to repeat.
func (i *Index) Migrate(ctx context.Context) error {
	return withTx(ctx, i.db, func(tx *sql.Tx) error {
		statements := []string{
			fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL, index_identity TEXT NOT NULL DEFAULT '')`, i.table),
			fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_document_id ON %s(document_id)`, i.table, i.table),
			fmt.Sprintf(`CREATE VIRTUAL TABLE IF NOT EXISTS %s USING fts5(id UNINDEXED, content, filename)`, i.ftsTable),
		}
		for _, statement := range statements {
			if _, err := tx.ExecContext(ctx, statement); err != nil {
				return fmt.Errorf("migrate sqlite index: %w", err)
			}
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`ALTER TABLE %s ADD COLUMN index_identity TEXT NOT NULL DEFAULT ''`, i.table)); err != nil && !strings.Contains(err.Error(), "duplicate column name") {
			return fmt.Errorf("migrate sqlite index identity: %w", err)
		}
		return nil
	})
}

func (i *Index) Upsert(ctx context.Context, items []*retrieval.Result) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return withTx(ctx, i.db, func(tx *sql.Tx) error {
		dimension, err := currentDimension(ctx, tx, i.table)
		if err != nil {
			return err
		}
		identity, identitySet, err := currentIdentity(ctx, tx, i.table)
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
			if !identitySet {
				identity, identitySet = item.IndexIdentity, true
			} else if item.IndexIdentity != identity {
				return fmt.Errorf("%w: index has %q, item %q has %q", indexcontract.ErrIdentityMismatch, identity, item.ID, item.IndexIdentity)
			}
			if len(item.Embedding) > 0 {
				if dimension == 0 {
					dimension = len(item.Embedding)
				} else if len(item.Embedding) != dimension {
					return fmt.Errorf("%w: index has %d dimensions, item %q has %d", indexcontract.ErrDimensionMismatch, dimension, item.ID, len(item.Embedding))
				}
			}
			metadata, err := json.Marshal(item.Metadata)
			if err != nil {
				return fmt.Errorf("sqlite index: encode metadata for %q: %w", item.ID, err)
			}
			embedding, err := json.Marshal(item.Embedding)
			if err != nil {
				return err
			}
			if _, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (id, document_id, filename, content, metadata, embedding, index_identity) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET document_id=excluded.document_id, filename=excluded.filename, content=excluded.content, metadata=excluded.metadata, embedding=excluded.embedding, index_identity=excluded.index_identity`, i.table), item.ID, item.DocumentID, item.Filename, item.Content, string(metadata), string(embedding), item.IndexIdentity); err != nil {
				return fmt.Errorf("sqlite index: upsert %q: %w", item.ID, err)
			}
			if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE id = ?`, i.ftsTable), item.ID); err != nil {
				return err
			}
			if _, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (id, content, filename) VALUES (?, ?, ?)`, i.ftsTable), item.ID, item.Content, item.Filename); err != nil {
				return err
			}
		}
		return nil
	})
}

func (i *Index) DeleteDocument(ctx context.Context, documentID string) error {
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
		return nil
	})
}

func (i *Index) Search(ctx context.Context, query indexcontract.IndexQuery) ([]*retrieval.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
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
	dimension, err := currentDimension(ctx, tx, i.table)
	if err != nil {
		return nil, err
	}
	identity, identitySet, err := currentIdentity(ctx, tx, i.table)
	if err != nil {
		return nil, err
	}
	if identitySet && identity != query.Identity {
		return nil, fmt.Errorf("%w: index has %q, query has %q", indexcontract.ErrIdentityMismatch, identity, query.Identity)
	}
	if dimension > 0 && len(query.Vector) > 0 && len(query.Vector) != dimension {
		return nil, fmt.Errorf("%w: index has %d dimensions, query has %d", indexcontract.ErrDimensionMismatch, dimension, len(query.Vector))
	}
	where, args, err := buildFilters(query.Filter)
	if err != nil {
		return nil, err
	}
	limit := 0
	if query.Limit > 0 {
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
	if limit > 0 {
		statement += " LIMIT ?"
		args = append(args, limit)
	}
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
		if err := json.Unmarshal([]byte(metadata), &item.Metadata); err != nil {
			return nil, err
		}
		if err := json.Unmarshal([]byte(embedding), &item.Embedding); err != nil {
			return nil, err
		}
		candidates = append(candidates, &item)
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

func currentIdentity(ctx context.Context, q queryer, table string) (string, bool, error) {
	var minimum, maximum string
	var count int
	err := q.QueryRowContext(ctx, fmt.Sprintf(`SELECT COALESCE(MIN(index_identity), ''), COALESCE(MAX(index_identity), ''), COUNT(*) FROM %s`, table)).Scan(&minimum, &maximum, &count)
	if err != nil {
		return "", false, fmt.Errorf("sqlite index: inspect identity: %w", err)
	}
	if minimum != maximum {
		return "", false, fmt.Errorf("%w: stored rows contain multiple identities", indexcontract.ErrIdentityMismatch)
	}
	return minimum, count > 0, nil
}

type queryer interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func currentDimension(ctx context.Context, q queryer, table string) (int, error) {
	var raw string
	err := q.QueryRowContext(ctx, fmt.Sprintf(`SELECT embedding FROM %s WHERE embedding != 'null' AND embedding != '[]' ORDER BY id LIMIT 1`, table)).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("sqlite index: inspect dimension: %w", err)
	}
	var vector []float64
	if err := json.Unmarshal([]byte(raw), &vector); err != nil {
		return 0, err
	}
	return len(vector), nil
}

func buildFilters(filters map[string]any) (string, []any, error) {
	keys := make([]string, 0, len(filters))
	for key := range filters {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	var clauses []string
	var args []any
	for _, key := range keys {
		value := filters[key]
		switch value.(type) {
		case nil, string, bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		default:
			return "", nil, fmt.Errorf("sqlite index: filter %q must be scalar", key)
		}
		switch key {
		case "id", "document_id", "filename":
			clauses = append(clauses, "r."+key+" = ?")
			args = append(args, value)
		default:
			clauses = append(clauses, "json_extract(r.metadata, ?) = ?")
			args = append(args, "$."+key, value)
		}
	}
	if len(clauses) == 0 {
		return "", args, nil
	}
	return " WHERE " + strings.Join(clauses, " AND "), args, nil
}

func withTx(ctx context.Context, db *sql.DB, fn func(*sql.Tx) error) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	if err := fn(tx); err != nil {
		return errors.Join(err, tx.Rollback())
	}
	return tx.Commit()
}

var _ indexcontract.Index = (*Index)(nil)

// Package postgres provides a PostgreSQL pgvector-backed Reliquary index.
package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/internal/indexutil"
	"github.com/dotcommander/reliquary/retrieval"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgvector "github.com/pgvector/pgvector-go"
)

const defaultTable = "reliquary_index"

// Config controls PostgreSQL index storage. Migrate must be called explicitly
// before the index is used.
type Config struct {
	Table string
}

// Index stores retrieval candidates in a caller-owned PostgreSQL pool.
type Index struct {
	pool        *pgxpool.Pool
	table       string
	quoted      string
	stateTable  string
	stateQuoted string
}

// New validates configuration without connecting or performing migrations.
func New(pool *pgxpool.Pool, cfg Config) (*Index, error) {
	if pool == nil {
		return nil, fmt.Errorf("postgres index pool is nil")
	}
	if cfg.Table == "" {
		cfg.Table = defaultTable
	}
	if err := validateIdentifier(cfg.Table); err != nil {
		return nil, fmt.Errorf("postgres index table: %w", err)
	}
	stateTable := boundedStateTableName(cfg.Table)
	return &Index{pool: pool, table: cfg.Table, quoted: (pgx.Identifier{cfg.Table}).Sanitize(), stateTable: stateTable, stateQuoted: (pgx.Identifier{stateTable}).Sanitize()}, nil
}

// Migrate idempotently creates retrieval-owned PostgreSQL schema. It is never
// called implicitly by New or data operations.
func (i *Index) Migrate(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := i.pool.Exec(ctx, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("enable pgvector: %w", err)
	}
	tx, err := i.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin postgres index migration: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	query := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		id text PRIMARY KEY,
		document_id text NOT NULL DEFAULT '',
		filename text NOT NULL DEFAULT '',
		content text NOT NULL DEFAULT '',
		metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
		embedding vector,
		index_identity text NOT NULL DEFAULT ''
	)`, i.quoted)
	if _, err := tx.Exec(ctx, query); err != nil {
		return fmt.Errorf("create postgres index table: %w", err)
	}
	if _, err := tx.Exec(ctx, fmt.Sprintf(`ALTER TABLE %s ADD COLUMN IF NOT EXISTS index_identity text NOT NULL DEFAULT ''`, i.quoted)); err != nil {
		return fmt.Errorf("add postgres index identity: %w", err)
	}
	documentIndex := (pgx.Identifier{boundedIndexName(i.table, "document_id")}).Sanitize()
	if _, err := tx.Exec(ctx, fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s ON %s (document_id)`, documentIndex, i.quoted)); err != nil {
		return fmt.Errorf("create document index: %w", err)
	}
	if _, err := tx.Exec(ctx, fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton), index_identity text, embedding_dimension integer CHECK (embedding_dimension IS NULL OR embedding_dimension > 0))`, i.stateQuoted)); err != nil {
		return fmt.Errorf("create postgres index state table: %w", err)
	}
	if err := i.backfillState(ctx, tx); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit postgres index migration: %w", err)
	}
	return nil
}

// Upsert transactionally inserts or overwrites results by ID.
func (i *Index) Upsert(ctx context.Context, items []*retrieval.Result) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	tx, err := i.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin postgres index upsert: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, fmt.Sprintf(`LOCK TABLE %s IN SHARE ROW EXCLUSIVE MODE`, i.quoted)); err != nil {
		return fmt.Errorf("lock postgres index identity: %w", err)
	}
	space, err := i.readState(ctx, tx)
	if err != nil {
		return err
	}
	for _, item := range items {
		if item == nil {
			continue
		}
		if item.ID == "" {
			return fmt.Errorf("postgres index item ID is empty")
		}
	}
	space, err = space.ValidateResults(items)
	if err != nil {
		return err
	}
	if err := i.writeState(ctx, tx, space); err != nil {
		return err
	}

	query := fmt.Sprintf(`INSERT INTO %s (id, document_id, filename, content, metadata, embedding, index_identity)
		VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
		ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, filename = EXCLUDED.filename,
		content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding, index_identity = EXCLUDED.index_identity`, i.quoted)
	for _, item := range items {
		if item == nil {
			continue
		}
		if err := upsertItem(ctx, tx, query, item); err != nil {
			return err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit postgres index upsert: %w", err)
	}
	return nil
}

// ReplaceDocuments atomically replaces complete document revisions. The table
// lock serializes retained-corpus dimension and identity discovery with all
// other writers.
func (i *Index) ReplaceDocuments(ctx context.Context, replacements []indexcontract.DocumentReplacement) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := indexutil.ValidateReplacements(replacements); err != nil {
		return err
	}
	tx, err := i.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin postgres index replacement: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, fmt.Sprintf(`LOCK TABLE %s IN SHARE ROW EXCLUSIVE MODE`, i.quoted)); err != nil {
		return fmt.Errorf("lock postgres index replacement: %w", err)
	}
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
		if _, err := tx.Exec(ctx, fmt.Sprintf(`DELETE FROM %s WHERE document_id = $1`, i.quoted), replacement.DocumentID); err != nil {
			return fmt.Errorf("delete postgres replacement document %q: %w", replacement.DocumentID, err)
		}
	}
	for _, replacement := range replacements {
		for _, item := range replacement.Results {
			if item == nil {
				continue
			}
			var retainedDocumentID string
			err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT document_id FROM %s WHERE id = $1`, i.quoted), item.ID).Scan(&retainedDocumentID)
			switch {
			case err == nil:
				return fmt.Errorf("%w: %q belongs to retained document %q", indexcontract.ErrResultIDConflict, item.ID, retainedDocumentID)
			case errors.Is(err, pgx.ErrNoRows):
			case err != nil:
				return fmt.Errorf("inspect postgres replacement result %q: %w", item.ID, err)
			}
		}
	}
	query := fmt.Sprintf(`INSERT INTO %s (id, document_id, filename, content, metadata, embedding, index_identity)
		VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
		ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, filename = EXCLUDED.filename,
		content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding, index_identity = EXCLUDED.index_identity`, i.quoted)
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
			if err := upsertItem(ctx, tx, query, item); err != nil {
				return err
			}
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit postgres index replacement: %w", err)
	}
	return nil
}

func upsertItem(ctx context.Context, tx pgx.Tx, query string, item *retrieval.Result) error {
	metadata, err := json.Marshal(item.Metadata)
	if err != nil {
		return fmt.Errorf("marshal metadata for %q: %w", item.ID, err)
	}
	var vector any
	if len(item.Embedding) > 0 {
		values := make([]float32, len(item.Embedding))
		for n, value := range item.Embedding {
			values[n] = float32(value)
		}
		vector = pgvector.NewVector(values)
	}
	if _, err := tx.Exec(ctx, query, item.ID, item.DocumentID, item.Filename, item.Content, metadata, vector, item.IndexIdentity); err != nil {
		return fmt.Errorf("upsert postgres index item %q: %w", item.ID, err)
	}
	return nil
}

// DeleteDocument transactionally removes all results for a document.
func (i *Index) DeleteDocument(ctx context.Context, documentID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := indexutil.ValidateDocumentID(documentID); err != nil {
		return err
	}
	tx, err := i.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin postgres index delete: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, fmt.Sprintf(`DELETE FROM %s WHERE document_id = $1`, i.quoted), documentID); err != nil {
		return fmt.Errorf("delete postgres index document %q: %w", documentID, err)
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit postgres index delete: %w", err)
	}
	return nil
}

// Reset destructively removes all indexed results.
func (i *Index) Reset(ctx context.Context) error {
	tx, err := i.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin postgres index reset: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, fmt.Sprintf(`DELETE FROM %s`, i.quoted)); err != nil {
		return fmt.Errorf("reset postgres index: %w", err)
	}
	if _, err := tx.Exec(ctx, fmt.Sprintf(`DELETE FROM %s`, i.stateQuoted)); err != nil {
		return fmt.Errorf("reset postgres index state: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit postgres index reset: %w", err)
	}
	return nil
}

// Search retrieves candidates in PostgreSQL and performs final Reliquary
// scoring over only that bounded candidate set.
func (i *Index) Search(ctx context.Context, query indexcontract.IndexQuery) ([]*retrieval.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := indexutil.ValidateFilter(query.Filter); err != nil {
		return nil, fmt.Errorf("postgres index: %w", err)
	}
	tx, err := i.pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
	if err != nil {
		return nil, fmt.Errorf("begin postgres index search: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	space, err := i.readState(ctx, tx)
	if err != nil {
		return nil, err
	}
	if err := space.ValidateQuery(query); err != nil {
		return nil, err
	}

	sqlQuery, args, err := i.searchQuery(query)
	if err != nil {
		return nil, err
	}
	rows, err := tx.Query(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("query postgres index: %w", err)
	}
	defer rows.Close()

	items := make([]*retrieval.Result, 0)
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var item retrieval.Result
		var metadata []byte
		var vector *pgvector.Vector
		if err := rows.Scan(&item.ID, &item.DocumentID, &item.Filename, &item.Content, &metadata, &vector, &item.IndexIdentity); err != nil {
			return nil, fmt.Errorf("scan postgres index result: %w", err)
		}
		if len(metadata) > 0 {
			decoder := json.NewDecoder(bytes.NewReader(metadata))
			decoder.UseNumber()
			if err := decoder.Decode(&item.Metadata); err != nil {
				return nil, fmt.Errorf("decode metadata for %q: %w", item.ID, err)
			}
		}
		if vector != nil {
			values := vector.Slice()
			item.Embedding = make([]float64, len(values))
			for n, value := range values {
				item.Embedding[n] = float64(value)
			}
		}
		items = append(items, &item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate postgres index results: %w", err)
	}
	rows.Close()
	if err := tx.Commit(ctx); err != nil {
		return nil, fmt.Errorf("commit postgres index search: %w", err)
	}
	return indexutil.Search(ctx, query, items)
}

type rowQueryer interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}

func (i *Index) readState(ctx context.Context, q rowQueryer) (indexutil.Space, error) {
	var identity *string
	var dimension *int
	err := q.QueryRow(ctx, fmt.Sprintf(`SELECT index_identity, embedding_dimension FROM %s WHERE singleton = true`, i.stateQuoted)).Scan(&identity, &dimension)
	if errors.Is(err, pgx.ErrNoRows) {
		return indexutil.Space{}, nil
	}
	if err != nil {
		return indexutil.Space{}, fmt.Errorf("read postgres index state: %w", err)
	}
	state := indexutil.Space{}
	if identity != nil {
		state.Identity, state.IdentitySet = *identity, true
	}
	if dimension != nil {
		state.Dimension = *dimension
	}
	return state, nil
}

func (i *Index) writeState(ctx context.Context, tx pgx.Tx, state indexutil.Space) error {
	if !state.IdentitySet && state.Dimension == 0 {
		return nil
	}
	var identity *string
	if state.IdentitySet {
		identity = &state.Identity
	}
	var dimension *int
	if state.Dimension > 0 {
		dimension = &state.Dimension
	}
	_, err := tx.Exec(ctx, fmt.Sprintf(`INSERT INTO %s (singleton, index_identity, embedding_dimension) VALUES (true, $1, $2) ON CONFLICT (singleton) DO UPDATE SET index_identity=EXCLUDED.index_identity, embedding_dimension=EXCLUDED.embedding_dimension`, i.stateQuoted), identity, dimension)
	if err != nil {
		return fmt.Errorf("write postgres index state: %w", err)
	}
	return nil
}

func (i *Index) backfillState(ctx context.Context, tx pgx.Tx) error {
	state, err := i.readState(ctx, tx)
	if err != nil {
		return err
	}
	var minimum, maximum string
	var count int
	if err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT COALESCE(MIN(index_identity), ''), COALESCE(MAX(index_identity), ''), COUNT(*) FROM %s`, i.quoted)).Scan(&minimum, &maximum, &count); err != nil {
		return fmt.Errorf("inspect postgres legacy identities: %w", err)
	}
	if minimum != maximum {
		return fmt.Errorf("%w: stored rows contain multiple identities", indexcontract.ErrIdentityMismatch)
	}
	legacy := indexutil.Space{}
	if count > 0 {
		legacy.Identity, legacy.IdentitySet = minimum, true
	}
	var minDimension, maxDimension *int
	if err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT MIN(vector_dims(embedding)), MAX(vector_dims(embedding)) FROM %s WHERE embedding IS NOT NULL`, i.quoted)).Scan(&minDimension, &maxDimension); err != nil {
		return fmt.Errorf("inspect postgres legacy dimensions: %w", err)
	}
	if minDimension != nil {
		if *minDimension != *maxDimension {
			return fmt.Errorf("%w: stored rows contain multiple dimensions", indexcontract.ErrDimensionMismatch)
		}
		legacy.Dimension = *minDimension
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

func (i *Index) searchQuery(query indexcontract.IndexQuery) (string, []any, error) {
	if err := indexutil.ValidateFilter(query.Filter); err != nil {
		return "", nil, fmt.Errorf("postgres index: %w", err)
	}
	args := make([]any, 0, len(query.Filter)+1)
	where := make([]string, 0, len(query.Filter)+1)
	keys := make([]string, 0, len(query.Filter))
	for key := range query.Filter {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		value := query.Filter[key]
		switch key {
		case "id", "document_id", "filename":
			if _, ok := value.(string); !ok {
				where = append(where, "FALSE")
				continue
			}
			args = append(args, value)
			where = append(where, fmt.Sprintf(`%s = $%d`, key, len(args)))
		default:
			encoded, err := json.Marshal(map[string]any{key: value})
			if err != nil {
				return "", nil, fmt.Errorf("encode postgres index filter %q: %w", key, err)
			}
			args = append(args, encoded)
			where = append(where, fmt.Sprintf(`metadata @> $%d::jsonb`, len(args)))
		}
	}

	order := `id ASC`
	if len(query.Vector) > 0 {
		values := slices.Clone(query.Vector)
		args = append(args, pgvector.NewVector(values))
		order = fmt.Sprintf(`embedding <=> $%d NULLS LAST, id ASC`, len(args))
	}
	statement := fmt.Sprintf(`SELECT id, document_id, filename, content, metadata, embedding, index_identity FROM %s`, i.quoted)
	if len(where) > 0 {
		statement += ` WHERE ` + strings.Join(where, ` AND `)
	}
	statement += ` ORDER BY ` + order
	if query.Limit > 0 {
		args = append(args, query.Limit)
		statement += fmt.Sprintf(` LIMIT $%d`, len(args))
	}
	return statement, args, nil
}

func validateIdentifier(name string) error {
	if name == "" || len(name) > 63 || strings.TrimSpace(name) != name {
		return fmt.Errorf("identifier must be non-blank and at most 63 bytes")
	}
	for n := range len(name) {
		char := name[n]
		if char >= 0x80 || !(char == '_' || char >= 'A' && char <= 'Z' || char >= 'a' && char <= 'z' || n > 0 && char >= '0' && char <= '9') {
			return fmt.Errorf("invalid identifier %q", name)
		}
	}
	return nil
}

func boundedIndexName(table, suffix string) string {
	return boundedIdentifier("idx_" + table + "_" + suffix)
}

func boundedStateTableName(table string) string {
	return boundedIdentifier(table + "_state")
}

func boundedIdentifier(name string) string {
	if len(name) <= 63 {
		return name
	}
	sum := sha256.Sum256([]byte(name))
	suffix := fmt.Sprintf("_%x", sum[:6])
	return name[:63-len(suffix)] + suffix
}

var _ indexcontract.Index = (*Index)(nil)

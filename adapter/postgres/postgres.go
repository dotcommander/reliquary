// Package postgres provides a PostgreSQL pgvector-backed Reliquary index.
package postgres

import (
	"context"
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
	pool   *pgxpool.Pool
	table  string
	quoted string
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
	return &Index{pool: pool, table: cfg.Table, quoted: (pgx.Identifier{cfg.Table}).Sanitize()}, nil
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
	query := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		id text PRIMARY KEY,
		document_id text NOT NULL DEFAULT '',
		filename text NOT NULL DEFAULT '',
		content text NOT NULL DEFAULT '',
		metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
		embedding vector,
		index_identity text NOT NULL DEFAULT ''
	)`, i.quoted)
	if _, err := i.pool.Exec(ctx, query); err != nil {
		return fmt.Errorf("create postgres index table: %w", err)
	}
	if _, err := i.pool.Exec(ctx, fmt.Sprintf(`ALTER TABLE %s ADD COLUMN IF NOT EXISTS index_identity text NOT NULL DEFAULT ''`, i.quoted)); err != nil {
		return fmt.Errorf("add postgres index identity: %w", err)
	}
	documentIndex := (pgx.Identifier{boundedIndexName(i.table, "document_id")}).Sanitize()
	if _, err := i.pool.Exec(ctx, fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s ON %s (document_id)`, documentIndex, i.quoted)); err != nil {
		return fmt.Errorf("create document index: %w", err)
	}
	return nil
}

// Upsert transactionally inserts or overwrites results by ID.
func (i *Index) Upsert(ctx context.Context, items []*retrieval.Result) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	dimension, err := i.embeddingDimension(ctx)
	if err != nil {
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
	identity, identitySet, err := i.indexIdentity(ctx, tx)
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
		if !identitySet {
			identity, identitySet = item.IndexIdentity, true
		} else if item.IndexIdentity != identity {
			return fmt.Errorf("%w: index has %q, item %q has %q", indexcontract.ErrIdentityMismatch, identity, item.ID, item.IndexIdentity)
		}
		if len(item.Embedding) > 0 {
			if dimension != 0 && dimension != len(item.Embedding) {
				return fmt.Errorf("%w: index has %d dimensions, item %q has %d", indexcontract.ErrDimensionMismatch, dimension, item.ID, len(item.Embedding))
			}
			dimension = len(item.Embedding)
		}
	}

	query := fmt.Sprintf(`INSERT INTO %s (id, document_id, filename, content, metadata, embedding, index_identity)
		VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
		ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, filename = EXCLUDED.filename,
		content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding, index_identity = EXCLUDED.index_identity`, i.quoted)
	for _, item := range items {
		if item == nil {
			continue
		}
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
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit postgres index upsert: %w", err)
	}
	return nil
}

// DeleteDocument transactionally removes all results for a document.
func (i *Index) DeleteDocument(ctx context.Context, documentID string) error {
	if err := ctx.Err(); err != nil {
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
	tx, err := i.pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead})
	if err != nil {
		return nil, fmt.Errorf("begin postgres index search: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	identity, identitySet, err := i.indexIdentity(ctx, tx)
	if err != nil {
		return nil, err
	}
	if identitySet && identity != query.Identity {
		return nil, fmt.Errorf("%w: index has %q, query has %q", indexcontract.ErrIdentityMismatch, identity, query.Identity)
	}
	if len(query.Vector) > 0 {
		dimension, err := i.embeddingDimensionWith(ctx, tx)
		if err != nil {
			return nil, err
		}
		if dimension != 0 && dimension != len(query.Vector) {
			return nil, fmt.Errorf("%w: index has %d dimensions, query has %d", indexcontract.ErrDimensionMismatch, dimension, len(query.Vector))
		}
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
			if err := json.Unmarshal(metadata, &item.Metadata); err != nil {
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

func (i *Index) indexIdentity(ctx context.Context, q rowQueryer) (string, bool, error) {
	var minimum, maximum string
	var count int
	query := fmt.Sprintf(`SELECT COALESCE(MIN(index_identity), ''), COALESCE(MAX(index_identity), ''), COUNT(*) FROM %s`, i.quoted)
	if err := q.QueryRow(ctx, query).Scan(&minimum, &maximum, &count); err != nil {
		return "", false, fmt.Errorf("inspect postgres index identity: %w", err)
	}
	if minimum != maximum {
		return "", false, fmt.Errorf("%w: stored rows contain multiple identities", indexcontract.ErrIdentityMismatch)
	}
	return minimum, count > 0, nil
}

func (i *Index) embeddingDimension(ctx context.Context) (int, error) {
	return i.embeddingDimensionWith(ctx, i.pool)
}

func (i *Index) embeddingDimensionWith(ctx context.Context, q rowQueryer) (int, error) {
	var dimension int
	query := fmt.Sprintf(`SELECT COALESCE(vector_dims(embedding), 0) FROM %s WHERE embedding IS NOT NULL LIMIT 1`, i.quoted)
	err := q.QueryRow(ctx, query).Scan(&dimension)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("inspect postgres index dimension: %w", err)
	}
	return dimension, nil
}

func (i *Index) searchQuery(query indexcontract.IndexQuery) (string, []any, error) {
	args := make([]any, 0, len(query.Filter)+1)
	where := make([]string, 0, len(query.Filter)+1)
	keys := make([]string, 0, len(query.Filter))
	for key := range query.Filter {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		value := query.Filter[key]
		if !isScalar(value) {
			return "", nil, fmt.Errorf("postgres index filter %q must be scalar", key)
		}
		switch key {
		case "id", "document_id", "filename":
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

func isScalar(value any) bool {
	switch value.(type) {
	case nil, bool, string, float32, float64, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	default:
		return false
	}
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
	name := "idx_" + table + "_" + suffix
	if len(name) <= 63 {
		return name
	}
	// Table is already validated and bounded; truncation remains deterministic.
	return name[:63]
}

var _ indexcontract.Index = (*Index)(nil)

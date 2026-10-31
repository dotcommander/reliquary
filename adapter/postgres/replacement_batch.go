package postgres

import (
	"context"
	"encoding/json"
	"fmt"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
	"github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"
)

// Limit both the number of IDs per lookup and queued statements per round trip.
const replacementBatchSize = 256

func (i *Index) checkRetainedIDs(ctx context.Context, tx pgx.Tx, items []*retrieval.Result) error {
	query := fmt.Sprintf(`SELECT id, document_id FROM %s WHERE id = ANY($1::text[]) ORDER BY array_position($1::text[], id) LIMIT 1`, i.quoted)
	for start := 0; start < len(items); start += replacementBatchSize {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(start+replacementBatchSize, len(items))
		ids := make([]string, 0, end-start)
		for _, item := range items[start:end] {
			if item != nil {
				ids = append(ids, item.ID)
			}
		}
		if len(ids) == 0 {
			continue
		}
		rows, err := tx.Query(ctx, query, ids)
		if err != nil {
			return fmt.Errorf("inspect postgres replacement IDs: %w", err)
		}
		for rows.Next() {
			var id, documentID string
			if err := rows.Scan(&id, &documentID); err != nil {
				rows.Close()
				return fmt.Errorf("scan postgres retained result: %w", err)
			}
			rows.Close()
			return fmt.Errorf("%w: %q belongs to retained document %q", indexcontract.ErrResultIDConflict, id, documentID)
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return fmt.Errorf("inspect postgres replacement IDs: %w", err)
		}
	}
	return ctx.Err()
}

func (i *Index) upsertQuery() string {
	return fmt.Sprintf(`INSERT INTO %s (id, document_id, filename, content, metadata, embedding, index_identity)
		VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
		ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, filename = EXCLUDED.filename,
		content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding, index_identity = EXCLUDED.index_identity`, i.quoted)
}

func itemArguments(item *retrieval.Result) ([]any, error) {
	metadata, err := json.Marshal(item.Metadata)
	if err != nil {
		return nil, fmt.Errorf("marshal metadata for %q: %w", item.ID, err)
	}
	var vector any
	if len(item.Embedding) > 0 {
		values := make([]float32, len(item.Embedding))
		for n, value := range item.Embedding {
			values[n] = float32(value)
		}
		vector = pgvector.NewVector(values)
	}
	return []any{item.ID, item.DocumentID, item.Filename, item.Content, metadata, vector, item.IndexIdentity}, nil
}

func upsertItem(ctx context.Context, tx pgx.Tx, query string, item *retrieval.Result) error {
	args, err := itemArguments(item)
	if err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, query, args...); err != nil {
		return fmt.Errorf("upsert postgres index item %q: %w", item.ID, err)
	}
	return nil
}

func (i *Index) writeReplacementItems(ctx context.Context, tx pgx.Tx, items []*retrieval.Result) error {
	query := i.upsertQuery()
	for start := 0; start < len(items); start += replacementBatchSize {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(start+replacementBatchSize, len(items))
		var batch pgx.Batch
		queued := make([]*retrieval.Result, 0, end-start)
		for _, item := range items[start:end] {
			if item == nil {
				continue
			}
			args, err := itemArguments(item)
			if err != nil {
				return err
			}
			batch.Queue(query, args...)
			queued = append(queued, item)
		}
		if batch.Len() == 0 {
			continue
		}
		results := tx.SendBatch(ctx, &batch)
		for _, item := range queued {
			if _, err := results.Exec(); err != nil {
				_ = results.Close()
				return fmt.Errorf("upsert postgres index item %q: %w", item.ID, err)
			}
		}
		if err := results.Close(); err != nil {
			return fmt.Errorf("close postgres replacement batch: %w", err)
		}
	}
	return ctx.Err()
}

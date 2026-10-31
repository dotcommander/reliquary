package sqlite

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

// Keep the IN list below SQLite's historical 999-variable limit.
const replacementLookupBatchSize = 256

// deleteReplacementFTS removes any prior FTS rows for incoming IDs, including
// orphan rows, in bounded statements. Document-level cleanup may already have
// removed some of these rows; all deletion remains in the replacement transaction.
func (i *Index) deleteReplacementFTS(ctx context.Context, tx *sql.Tx, items []*retrieval.Result) error {
	for start := 0; start < len(items); start += replacementLookupBatchSize {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(start+replacementLookupBatchSize, len(items))
		ids := make([]any, 0, end-start)
		for _, item := range items[start:end] {
			if item != nil {
				ids = append(ids, item.ID)
			}
		}
		if len(ids) == 0 {
			continue
		}
		query := fmt.Sprintf(`DELETE FROM %s WHERE id IN (%s)`, i.ftsTable, strings.TrimSuffix(strings.Repeat("?,", len(ids)), ","))
		if _, err := tx.ExecContext(ctx, query, ids...); err != nil {
			return fmt.Errorf("sqlite index: delete prior replacement FTS rows: %w", err)
		}
	}
	return ctx.Err()
}

// checkRetainedReplacementIDs runs after document deletion in the same transaction.
// Checking results in input order preserves which conflicting ID is reported.
func (i *Index) checkRetainedReplacementIDs(ctx context.Context, tx *sql.Tx, items []*retrieval.Result) error {
	for start := 0; start < len(items); start += replacementLookupBatchSize {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(start+replacementLookupBatchSize, len(items))
		ids := make([]any, 0, end-start)
		for _, item := range items[start:end] {
			if item != nil {
				ids = append(ids, item.ID)
			}
		}
		if len(ids) == 0 {
			continue
		}
		query := fmt.Sprintf(`SELECT id, document_id FROM %s WHERE id IN (%s)`, i.table, strings.TrimSuffix(strings.Repeat("?,", len(ids)), ","))
		rows, err := tx.QueryContext(ctx, query, ids...)
		if err != nil {
			return fmt.Errorf("sqlite index: inspect replacement IDs: %w", err)
		}
		retained := make(map[string]string)
		for rows.Next() {
			var id, documentID string
			if err := rows.Scan(&id, &documentID); err != nil {
				rows.Close()
				return fmt.Errorf("sqlite index: scan retained replacement ID: %w", err)
			}
			retained[id] = documentID
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return fmt.Errorf("sqlite index: inspect replacement IDs: %w", err)
		}
		for _, item := range items[start:end] {
			if item == nil {
				continue
			}
			if documentID, exists := retained[item.ID]; exists {
				return fmt.Errorf("%w: %q belongs to retained document %q", indexcontract.ErrResultIDConflict, item.ID, documentID)
			}
		}
	}
	return ctx.Err()
}

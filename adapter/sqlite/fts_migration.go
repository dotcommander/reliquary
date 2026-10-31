package sqlite

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

// Migrate creates only the retrieval-owned schema and is safe to repeat.
func (i *Index) Migrate(ctx context.Context) error {
	return i.migrateFTS(ctx, false)
}

func (i *Index) migrateFTS(ctx context.Context, repair bool) error {
	return withTx(ctx, i.db, func(tx *sql.Tx) error {
		var ftsExisted bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)`, i.ftsTable).Scan(&ftsExisted); err != nil {
			return fmt.Errorf("migrate sqlite FTS schema: %w", err)
		}
		statements := []string{
			fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, filename TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL, index_identity TEXT NOT NULL DEFAULT '')`, i.table),
			fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_document_id ON %s(document_id)`, i.table, i.table),
			fmt.Sprintf(`CREATE VIRTUAL TABLE IF NOT EXISTS %s USING fts5(id UNINDEXED, content, filename)`, i.ftsTable),
			fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), index_identity TEXT, embedding_dimension INTEGER CHECK (embedding_dimension IS NULL OR embedding_dimension > 0), fts_version INTEGER NOT NULL DEFAULT 0)`, i.stateTable),
		}
		for _, statement := range statements {
			if _, err := tx.ExecContext(ctx, statement); err != nil {
				return fmt.Errorf("migrate sqlite index: %w", err)
			}
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`ALTER TABLE %s ADD COLUMN index_identity TEXT NOT NULL DEFAULT ''`, i.table)); err != nil && !strings.Contains(err.Error(), "duplicate column name") {
			return fmt.Errorf("migrate sqlite index identity: %w", err)
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`ALTER TABLE %s ADD COLUMN fts_version INTEGER NOT NULL DEFAULT 0`, i.stateTable)); err != nil && !strings.Contains(err.Error(), "duplicate column name") {
			return fmt.Errorf("migrate sqlite FTS version: %w", err)
		}
		// Continue validating identity/dimension drift even when the FTS
		// generation is already current.
		if err := i.backfillState(ctx, tx); err != nil {
			return err
		}
		version, err := i.ftsVersion(ctx, tx)
		if err != nil {
			return err
		}
		if version == currentFTSVersion && ftsExisted && !repair {
			return nil
		}
		return i.rebuildFTS(ctx, tx)
	})
}

// currentFTSVersion records when this adapter last rebuilt FTS from the base table.
// External writes bypassing the adapter invalidate this marker; RepairFTS
// explicitly restores parity after such writes.
const currentFTSVersion = 1

func (i *Index) ftsVersion(ctx context.Context, tx *sql.Tx) (int, error) {
	var version int
	err := tx.QueryRowContext(ctx, fmt.Sprintf(`SELECT fts_version FROM %s WHERE singleton = 1`, i.stateTable)).Scan(&version)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("sqlite index: read FTS version: %w", err)
	}
	if version > currentFTSVersion {
		return 0, fmt.Errorf("sqlite index: unsupported FTS version %d", version)
	}
	return version, nil
}

func (i *Index) rebuildFTS(ctx context.Context, tx *sql.Tx) error {
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s`, i.ftsTable)); err != nil {
		return fmt.Errorf("rebuild sqlite FTS index: %w", err)
	}
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (id, content, filename) SELECT id, content, filename FROM %s`, i.ftsTable, i.table)); err != nil {
		return fmt.Errorf("repopulate sqlite FTS index: %w", err)
	}
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (singleton, fts_version) VALUES (1, ?) ON CONFLICT(singleton) DO UPDATE SET fts_version=excluded.fts_version`, i.stateTable), currentFTSVersion); err != nil {
		return fmt.Errorf("sqlite index: record FTS version: %w", err)
	}
	return nil
}

// RepairFTS explicitly rebuilds the search table from the base table. Use it
// after external writes to either table; ordinary repeat Migrate skips the
// full rebuild on a current generation. This operation is transactional.
func (i *Index) RepairFTS(ctx context.Context) error {
	return i.migrateFTS(ctx, true)
}

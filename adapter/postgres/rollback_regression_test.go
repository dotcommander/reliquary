package postgres

import (
	"context"
	"errors"
	"testing"
	"time"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

type fixturePool struct {
	tx       pgx.Tx
	beginCtx context.Context
}

func (p *fixturePool) Begin(ctx context.Context) (pgx.Tx, error) {
	p.beginCtx = ctx
	return p.tx, nil
}

func (p *fixturePool) BeginTx(ctx context.Context, _ pgx.TxOptions) (pgx.Tx, error) {
	return p.Begin(ctx)
}

func (p *fixturePool) Exec(context.Context, string, ...any) (pgconn.CommandTag, error) {
	return pgconn.CommandTag{}, nil
}

type fixtureTx struct {
	pgx.Tx   // Unexpected transaction methods panic rather than silently succeeding.
	exec     func(context.Context, string, ...any) (pgconn.CommandTag, error)
	queryRow func(context.Context, string, ...any) pgx.Row
	query    func(context.Context, string, ...any) (pgx.Rows, error)
	rollback func(context.Context) error
	commit   func(context.Context) error
}

func (tx *fixtureTx) Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error) {
	return tx.exec(ctx, sql, args...)
}
func (tx *fixtureTx) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	return tx.queryRow(ctx, sql, args...)
}
func (tx *fixtureTx) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	return tx.query(ctx, sql, args...)
}
func (tx *fixtureTx) Rollback(ctx context.Context) error { return tx.rollback(ctx) }
func (tx *fixtureTx) Commit(ctx context.Context) error   { return tx.commit(ctx) }

type fixtureRow func(...any) error

func (r fixtureRow) Scan(dest ...any) error { return r(dest...) }

type rollbackValueKey struct{}

func TestRollbackAfterOperationCancellation(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		call func(context.Context, *Index) error
	}{
		{"migrate", func(ctx context.Context, idx *Index) error { return idx.Migrate(ctx) }},
		{"upsert", func(ctx context.Context, idx *Index) error { return idx.Upsert(ctx, []*retrieval.Result{{ID: "a"}}) }},
		{"replace", func(ctx context.Context, idx *Index) error {
			return idx.ReplaceDocuments(ctx, []indexcontract.DocumentReplacement{{DocumentID: "doc"}})
		}},
		{"delete", func(ctx context.Context, idx *Index) error { return idx.DeleteDocument(ctx, "doc") }},
		{"reset", func(ctx context.Context, idx *Index) error { return idx.Reset(ctx) }},
		{"search", func(ctx context.Context, idx *Index) error {
			_, err := idx.Search(ctx, indexcontract.IndexQuery{})
			return err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ctx, cancel := context.WithTimeout(context.WithValue(t.Context(), rollbackValueKey{}, "request"), time.Second)
			t.Cleanup(cancel)
			primary := errors.New("operation failed")
			cleanupError := errors.New("cleanup failed")
			var operationCtx, cleanupCtx context.Context
			calls := 0
			fail := func(c context.Context) error {
				operationCtx = c
				cancel()
				return primary
			}
			tx := &fixtureTx{
				exec: func(c context.Context, _ string, _ ...any) (pgconn.CommandTag, error) {
					return pgconn.CommandTag{}, fail(c)
				},
				queryRow: func(c context.Context, _ string, _ ...any) pgx.Row {
					return fixtureRow(func(...any) error { return fail(c) })
				},
				rollback: func(c context.Context) error {
					calls++
					cleanupCtx = c
					if c.Err() != nil {
						t.Errorf("rollback received canceled context: %v", c.Err())
					}
					if c.Value(rollbackValueKey{}) != "request" {
						t.Error("rollback lost request values")
					}
					deadline, ok := c.Deadline()
					if remaining := time.Until(deadline); !ok || remaining <= 0 || remaining > rollbackTimeout {
						t.Errorf("cleanup deadline = %v, present = %v", deadline, ok)
					}
					return cleanupError
				},
				commit: func(context.Context) error { t.Error("commit after operation failure"); return nil },
			}
			pool := &fixturePool{tx: tx}
			idx := &Index{pool: pool, quoted: `"items"`, stateQuoted: `"state"`}
			err := tc.call(ctx, idx)
			if !errors.Is(err, primary) || errors.Is(err, cleanupError) {
				t.Fatalf("operation error = %v", err)
			}
			if pool.beginCtx != ctx || operationCtx != ctx {
				t.Fatal("operation context was replaced")
			}
			if calls != 1 {
				t.Fatalf("rollback calls = %d, want 1", calls)
			}
			if cleanupCtx.Err() != context.Canceled {
				t.Fatalf("cleanup resources remain live: %v", cleanupCtx.Err())
			}
		})
	}
}

package postgres

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

type retainedRows struct {
	pgx.Rows
	id, documentID string
	seen, closed   bool
	err            error
}

func (r *retainedRows) Next() bool {
	if r.id == "" || r.seen {
		return false
	}
	r.seen = true
	return true
}
func (r *retainedRows) Scan(dest ...any) error {
	*dest[0].(*string) = r.id
	*dest[1].(*string) = r.documentID
	return nil
}
func (r *retainedRows) Err() error { return r.err }
func (r *retainedRows) Close()     { r.closed = true }

type replacementBatchResults struct {
	pgx.BatchResults
	remaining int
	failAt    int
	failure   error
	closeErr  error
	calls     int
	closed    bool
}

func (r *replacementBatchResults) Exec() (pgconn.CommandTag, error) {
	r.calls++
	if r.calls == r.failAt {
		return pgconn.CommandTag{}, r.failure
	}
	if r.calls > r.remaining {
		panic("read past queued batch")
	}
	return pgconn.CommandTag{}, nil
}
func (r *replacementBatchResults) Close() error { r.closed = true; return r.closeErr }

func TestReplaceDocumentsBatchesRetainedChecksAndWrites(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	items := make([]*retrieval.Result, replacementBatchSize+1)
	for n := range items {
		items[n] = &retrieval.Result{ID: fmt.Sprintf("id-%d", n), DocumentID: "doc"}
	}
	var checked, written, deleted int
	var batches []*replacementBatchResults
	var rows []*retainedRows
	committed := false
	tx := &fixtureTx{
		exec: func(c context.Context, sql string, _ ...any) (pgconn.CommandTag, error) {
			if c != ctx {
				t.Error("exec context changed")
			}
			if strings.Contains(sql, "DELETE FROM") {
				deleted++
			}
			return pgconn.CommandTag{}, nil
		},
		queryRow: func(c context.Context, _ string, _ ...any) pgx.Row {
			if c != ctx {
				t.Error("state context changed")
			}
			return fixtureRow(func(...any) error { return pgx.ErrNoRows })
		},
		query: func(c context.Context, sql string, args ...any) (pgx.Rows, error) {
			if c != ctx || !strings.Contains(sql, "ANY($1::text[])") || !strings.Contains(sql, "ORDER BY array_position($1::text[], id) LIMIT 1") {
				t.Error("unbounded, unordered, or wrong-context collision query")
			}
			ids := args[0].([]string)
			if len(ids) > replacementBatchSize {
				t.Fatalf("unbounded ID query: %d", len(ids))
			}
			checked += len(ids)
			r := &retainedRows{}
			rows = append(rows, r)
			return r, nil
		},
		commit: func(c context.Context) error {
			if c != ctx {
				t.Error("commit context changed")
			}
			committed = true
			return nil
		},
		rollback: func(context.Context) error { return pgx.ErrTxClosed },
	}
	// The embedded transaction implements the remaining pgx methods; only
	// SendBatch is supplied by this test wrapper.
	wrapped := &batchedFixtureTx{fixtureTx: tx, send: func(c context.Context, batch *pgx.Batch) pgx.BatchResults {
		if c != ctx || batch.Len() > replacementBatchSize || batch.Len() == 0 {
			t.Error("unbounded or wrong-context write batch")
		}
		if got := batch.QueuedQueries[0].Arguments[0]; got != fmt.Sprintf("id-%d", written) {
			t.Errorf("batch starts at %v, want id-%d", got, written)
		}
		written += batch.Len()
		r := &replacementBatchResults{remaining: batch.Len()}
		batches = append(batches, r)
		return r
	}}
	idx := &Index{pool: &fixturePool{tx: wrapped}, quoted: `"items"`, stateQuoted: `"state"`}
	err := idx.ReplaceDocuments(ctx, []indexcontract.DocumentReplacement{{DocumentID: "doc", Results: items}})
	if err != nil {
		t.Fatal(err)
	}
	if !committed || deleted != 1 || checked != len(items) || written != len(items) || len(batches) != 2 || len(rows) != 2 {
		t.Fatalf("commit=%t deletes=%d checks=%d writes=%d batches=%d rows=%d", committed, deleted, checked, written, len(batches), len(rows))
	}
	for _, r := range rows {
		if !r.closed {
			t.Error("retained rows not closed")
		}
	}
	for _, r := range batches {
		if !r.closed || r.calls != r.remaining {
			t.Error("batch not drained and closed")
		}
	}
}

type batchedFixtureTx struct {
	*fixtureTx
	send func(context.Context, *pgx.Batch) pgx.BatchResults
}

func (tx *batchedFixtureTx) SendBatch(ctx context.Context, batch *pgx.Batch) pgx.BatchResults {
	return tx.send(ctx, batch)
}

func TestReplaceDocumentsBatchErrorsRollback(t *testing.T) {
	t.Parallel()
	failure := errors.New("database failure")
	for _, tc := range []struct {
		name     string
		rows     *retainedRows
		queryErr error
		writeErr error
		closeErr error
		want     error
	}{
		{name: "retained collision", rows: &retainedRows{id: "id", documentID: "old"}, want: indexcontract.ErrResultIDConflict},
		{name: "lookup failure", queryErr: failure, want: failure},
		{name: "iterate failure", rows: &retainedRows{err: failure}, want: failure},
		{name: "write failure", writeErr: failure, want: failure},
		{name: "close failure", closeErr: failure, want: failure},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rollbacks := 0
			committed := false
			var batch *replacementBatchResults
			tx := &batchedFixtureTx{fixtureTx: &fixtureTx{
				exec: func(context.Context, string, ...any) (pgconn.CommandTag, error) { return pgconn.CommandTag{}, nil },
				queryRow: func(context.Context, string, ...any) pgx.Row {
					return fixtureRow(func(...any) error { return pgx.ErrNoRows })
				},
				query:    func(context.Context, string, ...any) (pgx.Rows, error) { return tc.rows, tc.queryErr },
				commit:   func(context.Context) error { committed = true; return nil },
				rollback: func(context.Context) error { rollbacks++; return nil },
			}, send: func(_ context.Context, b *pgx.Batch) pgx.BatchResults {
				batch = &replacementBatchResults{remaining: b.Len(), failAt: 1, failure: tc.writeErr, closeErr: tc.closeErr}
				return batch
			}}
			if tc.rows == nil && tc.queryErr == nil {
				tc.rows = &retainedRows{}
			}
			idx := &Index{pool: &fixturePool{tx: tx}, quoted: `"items"`, stateQuoted: `"state"`}
			err := idx.ReplaceDocuments(t.Context(), []indexcontract.DocumentReplacement{{DocumentID: "doc", Results: []*retrieval.Result{{ID: "id", DocumentID: "doc"}}}})
			if !errors.Is(err, tc.want) || committed || rollbacks != 1 {
				t.Fatalf("err=%v committed=%t rollbacks=%d", err, committed, rollbacks)
			}
			if tc.rows != nil && !tc.rows.closed {
				t.Error("lookup rows not closed")
			}
			if batch != nil && !batch.closed {
				t.Error("failed batch not closed")
			}
			if errors.Is(err, indexcontract.ErrResultIDConflict) && batch != nil {
				t.Error("wrote before detecting retained collision")
			}
		})
	}
}

func TestReplaceDocumentsCancellationBeforeCommit(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	committed, rolledBack := false, false
	tx := &batchedFixtureTx{fixtureTx: &fixtureTx{
		exec: func(context.Context, string, ...any) (pgconn.CommandTag, error) { return pgconn.CommandTag{}, nil },
		queryRow: func(context.Context, string, ...any) pgx.Row {
			return fixtureRow(func(...any) error { return pgx.ErrNoRows })
		},
		query:    func(context.Context, string, ...any) (pgx.Rows, error) { return &retainedRows{}, nil },
		commit:   func(context.Context) error { committed = true; return nil },
		rollback: func(context.Context) error { rolledBack = true; return nil },
	}, send: func(_ context.Context, b *pgx.Batch) pgx.BatchResults {
		cancel()
		return &replacementBatchResults{remaining: b.Len()}
	}}
	idx := &Index{pool: &fixturePool{tx: tx}, quoted: `"items"`, stateQuoted: `"state"`}
	err := idx.ReplaceDocuments(ctx, []indexcontract.DocumentReplacement{{DocumentID: "doc", Results: []*retrieval.Result{{ID: "id", DocumentID: "doc"}}}})
	if !errors.Is(err, context.Canceled) || committed || !rolledBack {
		t.Fatalf("err=%v committed=%t rolledBack=%t", err, committed, rolledBack)
	}
}

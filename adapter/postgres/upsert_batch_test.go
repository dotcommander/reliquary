package postgres

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/dotcommander/reliquary/retrieval"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestUpsertBatchesInInputOrderWithinOneTransaction(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	items := make([]*retrieval.Result, replacementBatchSize+2)
	for n := range items {
		items[n] = &retrieval.Result{ID: fmt.Sprintf("id-%d", n)}
	}
	items[1] = nil
	items[replacementBatchSize+1].ID = "id-0" // Duplicate IDs retain last-write-wins order.
	var batches []*replacementBatchResults
	var queuedIDs []string
	committed, rolledBack := false, false
	tx := &batchedFixtureTx{fixtureTx: &fixtureTx{
		exec: func(c context.Context, statement string, _ ...any) (pgconn.CommandTag, error) {
			if c != ctx || !strings.Contains(statement, "LOCK TABLE") && !strings.Contains(statement, "INSERT INTO") {
				t.Errorf("unexpected statement/context: %q", statement)
			}
			return pgconn.CommandTag{}, nil
		},
		queryRow: func(c context.Context, _ string, _ ...any) pgx.Row {
			if c != ctx {
				t.Error("state context changed")
			}
			return fixtureRow(func(...any) error { return pgx.ErrNoRows })
		},
		commit:   func(c context.Context) error { committed = c == ctx; return nil },
		rollback: func(context.Context) error { rolledBack = true; return pgx.ErrTxClosed },
	}, send: func(c context.Context, batch *pgx.Batch) pgx.BatchResults {
		if c != ctx || batch.Len() > replacementBatchSize || batch.Len() == 0 {
			t.Errorf("invalid batch length/context: %d", batch.Len())
		}
		for _, query := range batch.QueuedQueries {
			queuedIDs = append(queuedIDs, query.Arguments[0].(string))
		}
		r := &replacementBatchResults{remaining: batch.Len()}
		batches = append(batches, r)
		return r
	}}
	idx := &Index{pool: &fixturePool{tx: tx}, quoted: `"items"`, stateQuoted: `"state"`}
	if err := idx.Upsert(ctx, items); err != nil {
		t.Fatal(err)
	}
	// Upsert always attempts deferred rollback; pgx returns ErrTxClosed after commit.
	if !committed || !rolledBack || len(batches) != 2 || len(queuedIDs) != len(items)-1 {
		t.Fatalf("commit=%t rollback=%t batches=%d queued=%d", committed, rolledBack, len(batches), len(queuedIDs))
	}
	for _, batch := range batches {
		if !batch.closed || batch.calls != batch.remaining {
			t.Error("batch not drained and closed")
		}
	}
	if queuedIDs[0] != "id-0" || queuedIDs[1] != "id-2" || queuedIDs[len(queuedIDs)-1] != "id-0" {
		t.Fatalf("batch order = %v", queuedIDs)
	}
}

func TestUpsertBatchFailureRollsBack(t *testing.T) {
	t.Parallel()
	failure := errors.New("batch write failed")
	for _, tc := range []struct {
		name, id           string
		writeErr, closeErr error
	}{
		{name: "write", id: "second", writeErr: failure},
		{name: "close", closeErr: failure},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			committed, rolledBack, closed := false, false, false
			tx := &batchedFixtureTx{fixtureTx: &fixtureTx{
				exec: func(context.Context, string, ...any) (pgconn.CommandTag, error) { return pgconn.CommandTag{}, nil },
				queryRow: func(context.Context, string, ...any) pgx.Row {
					return fixtureRow(func(...any) error { return pgx.ErrNoRows })
				},
				commit:   func(context.Context) error { committed = true; return nil },
				rollback: func(context.Context) error { rolledBack = true; return nil },
			}, send: func(_ context.Context, batch *pgx.Batch) pgx.BatchResults {
				return &batchResultsWithClose{replacementBatchResults: &replacementBatchResults{remaining: batch.Len(), failAt: 2, failure: tc.writeErr, closeErr: tc.closeErr}, closed: &closed}
			}}
			idx := &Index{pool: &fixturePool{tx: tx}, quoted: `"items"`, stateQuoted: `"state"`}
			err := idx.Upsert(t.Context(), []*retrieval.Result{{ID: "first"}, {ID: "second"}})
			if !errors.Is(err, failure) || !rolledBack || committed || !closed {
				t.Fatalf("err=%v rollback=%t commit=%t closed=%t", err, rolledBack, committed, closed)
			}
			if tc.id != "" && !strings.Contains(err.Error(), tc.id) {
				t.Fatalf("error %q missing failed ID %q", err, tc.id)
			}
		})
	}
}

type batchResultsWithClose struct {
	*replacementBatchResults
	closed *bool
}

func (r *batchResultsWithClose) Close() error {
	*r.closed = true
	return r.replacementBatchResults.Close()
}

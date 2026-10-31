package postgres

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/jackc/pgx/v5"
)

func TestDeleteBatchedDocumentsBoundsAndOrdering(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	replacements := make([]indexcontract.DocumentReplacement, replacementBatchSize+1)
	for n := range replacements {
		replacements[n].DocumentID = fmt.Sprintf("doc-%d", n)
	}
	var sent, calls int
	var results []*replacementBatchResults
	tx := &batchedFixtureTx{send: func(c context.Context, batch *pgx.Batch) pgx.BatchResults {
		if c != ctx || batch.Len() == 0 || batch.Len() > replacementBatchSize {
			t.Error("delete batch has wrong context or size")
		}
		for _, queued := range batch.QueuedQueries {
			if !strings.Contains(queued.SQL, `DELETE FROM "items" WHERE document_id = $1`) || queued.Arguments[0] != replacements[sent].DocumentID {
				t.Errorf("delete %d: query=%q args=%v", sent, queued.SQL, queued.Arguments)
			}
			sent++
		}
		calls++
		r := &replacementBatchResults{remaining: batch.Len()}
		results = append(results, r)
		return r
	}}
	idx := &Index{quoted: `"items"`}
	if err := idx.deleteBatchedDocuments(ctx, tx, replacements); err != nil {
		t.Fatal(err)
	}
	if sent != len(replacements) || calls != 2 {
		t.Fatalf("sent=%d calls=%d", sent, calls)
	}
	for _, r := range results {
		if !r.closed || r.calls != r.remaining {
			t.Errorf("batch closed=%t calls=%d remaining=%d", r.closed, r.calls, r.remaining)
		}
	}
}

func TestDeleteBatchedDocumentsFailureIdentifiesDocument(t *testing.T) {
	t.Parallel()
	failure := errors.New("delete failed")
	var sent int
	var result *replacementBatchResults
	tx := &batchedFixtureTx{send: func(_ context.Context, batch *pgx.Batch) pgx.BatchResults {
		sent++
		result = &replacementBatchResults{remaining: batch.Len(), failAt: 2, failure: failure}
		return result
	}}
	idx := &Index{quoted: `"items"`}
	replacements := []indexcontract.DocumentReplacement{{DocumentID: "first"}, {DocumentID: "second"}, {DocumentID: "third"}}
	err := idx.deleteBatchedDocuments(t.Context(), tx, replacements)
	if !errors.Is(err, failure) || !strings.Contains(err.Error(), `"second"`) || sent != 1 || result == nil || !result.closed {
		t.Fatalf("err=%v sent=%d", err, sent)
	}
}

func TestDeleteBatchedDocumentsCancellationStopsNextBatch(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	var sent int
	tx := &batchedFixtureTx{send: func(_ context.Context, batch *pgx.Batch) pgx.BatchResults {
		sent++
		cancel()
		return &replacementBatchResults{remaining: batch.Len()}
	}}
	replacements := make([]indexcontract.DocumentReplacement, replacementBatchSize+1)
	for n := range replacements {
		replacements[n].DocumentID = fmt.Sprintf("doc-%d", n)
	}
	idx := &Index{quoted: `"items"`}
	if err := idx.deleteBatchedDocuments(ctx, tx, replacements); !errors.Is(err, context.Canceled) || sent != 1 {
		t.Fatalf("err=%v sent=%d", err, sent)
	}
}

package indexsink

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/pipeline/ingest"
	"github.com/dotcommander/reliquary/retrieval"
)

type fakeIndex struct {
	upserts [][]*retrieval.Result
	err     error
}

func (f *fakeIndex) Upsert(_ context.Context, items []*retrieval.Result) error {
	f.upserts = append(f.upserts, items)
	return f.err
}

func (f *fakeIndex) DeleteDocument(_ context.Context, _ string) error { return nil }

func (f *fakeIndex) Search(_ context.Context, _ index.IndexQuery) ([]*retrieval.Result, error) {
	return nil, nil
}

func TestSinkWrite_HappyPath(t *testing.T) {
	t.Parallel()
	fi := &fakeIndex{}
	sink := NewSink(fi)

	batch := ingest.Batch[*retrieval.Result]{
		Records: []ingest.Record[*retrieval.Result]{
			{ID: "a", Payload: &retrieval.Result{ID: "a"}},
			{ID: "b", Payload: nil},
			{ID: "c", Payload: &retrieval.Result{ID: "c"}},
		},
		Cursor: ingest.Cursor{Source: "s3", Token: "tok"},
	}

	report, err := sink.Write(context.Background(), batch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if report.Written != 2 {
		t.Errorf("Written = %d, want 2", report.Written)
	}
	if report.Skipped != 1 {
		t.Errorf("Skipped = %d, want 1", report.Skipped)
	}
	if report.Cursor != batch.Cursor {
		t.Errorf("Cursor = %+v, want %+v", report.Cursor, batch.Cursor)
	}
	if len(fi.upserts) != 1 || len(fi.upserts[0]) != 2 {
		t.Errorf("Upsert calls = %v, want one call with 2 items", fi.upserts)
	}
}

func TestSinkWrite_EmptyBatchDoesNotUpsert(t *testing.T) {
	t.Parallel()
	fi := &fakeIndex{}
	sink := NewSink(fi)

	batch := ingest.Batch[*retrieval.Result]{
		Records: []ingest.Record[*retrieval.Result]{
			{ID: "a", Payload: nil},
		},
		Cursor: ingest.Cursor{Source: "s3"},
	}

	report, err := sink.Write(context.Background(), batch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if report.Written != 0 {
		t.Errorf("Written = %d, want 0", report.Written)
	}
	if report.Skipped != 1 {
		t.Errorf("Skipped = %d, want 1", report.Skipped)
	}
	if len(fi.upserts) != 0 {
		t.Errorf("Upsert called %d times, want 0", len(fi.upserts))
	}
}

func TestSinkWrite_UpsertErrorAborts(t *testing.T) {
	t.Parallel()
	wantErr := errors.New("boom")
	fi := &fakeIndex{err: wantErr}
	sink := NewSink(fi)

	batch := ingest.Batch[*retrieval.Result]{
		Records: []ingest.Record[*retrieval.Result]{
			{ID: "a", Payload: &retrieval.Result{ID: "a"}},
			{ID: "b", Payload: &retrieval.Result{ID: "b"}},
		},
		Cursor: ingest.Cursor{Source: "s3"},
	}

	report, err := sink.Write(context.Background(), batch)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
	if report.Errors != 2 {
		t.Errorf("Errors = %d, want 2", report.Errors)
	}
	if report.Written != 0 {
		t.Errorf("Written = %d, want 0 on error", report.Written)
	}
}

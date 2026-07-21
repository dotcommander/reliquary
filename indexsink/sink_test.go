package indexsink

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/embed"
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

func (f *fakeIndex) ReplaceDocuments(_ context.Context, _ []index.DocumentReplacement) error {
	return nil
}

func (f *fakeIndex) DeleteDocument(_ context.Context, _ string) error { return nil }

func (f *fakeIndex) Search(_ context.Context, _ index.IndexQuery) ([]*retrieval.Result, error) {
	return nil, nil
}

func TestSinkWrite_HappyPath(t *testing.T) {
	t.Parallel()
	fi := &fakeIndex{}
	sink, err := NewSink(fi, Config{IndexIdentity: "test-space"})
	if err != nil {
		t.Fatalf("NewSink: %v", err)
	}

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
	if report.Read != 3 {
		t.Errorf("Read = %d, want 3", report.Read)
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
	sink, err := NewSink(fi, Config{IndexIdentity: "test-space"})
	if err != nil {
		t.Fatalf("NewSink: %v", err)
	}

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
	if report.Read != 1 {
		t.Errorf("Read = %d, want 1", report.Read)
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
	sink, err := NewSink(fi, Config{IndexIdentity: "test-space"})
	if err != nil {
		t.Fatalf("NewSink: %v", err)
	}

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
	if report.Read != 2 {
		t.Errorf("Read = %d, want 2", report.Read)
	}
	if report.Written != 0 {
		t.Errorf("Written = %d, want 0 on error", report.Written)
	}
}

func TestNewSink_ValidatesConfiguration(t *testing.T) {
	t.Parallel()

	var typedNil *fakeIndex
	tests := []struct {
		name   string
		idx    index.Index
		config Config
		want   error
	}{
		{name: "nil index", config: Config{IndexIdentity: "space"}, want: ErrNilIndex},
		{name: "typed nil index", idx: typedNil, config: Config{IndexIdentity: "space"}, want: ErrNilIndex},
		{name: "empty identity", idx: &fakeIndex{}, want: ErrInvalidIndexIdentity},
		{name: "blank identity", idx: &fakeIndex{}, config: Config{IndexIdentity: " \t"}, want: ErrInvalidIndexIdentity},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			sink, err := NewSink(tt.idx, tt.config)
			if sink != nil {
				t.Errorf("sink = %#v, want nil", sink)
			}
			if !errors.Is(err, tt.want) {
				t.Errorf("err = %v, want errors.Is(_, %v)", err, tt.want)
			}
		})
	}
}

func TestSinkWrite_EnforcesIdentityWithoutMutatingCaller(t *testing.T) {
	t.Parallel()
	fi := &fakeIndex{}
	sink, err := NewSink(fi, Config{IndexIdentity: "test-space"})
	if err != nil {
		t.Fatalf("NewSink: %v", err)
	}
	missing := &retrieval.Result{ID: "missing"}
	matching := &retrieval.Result{ID: "matching", IndexIdentity: "test-space"}

	report, err := sink.Write(context.Background(), ingest.Batch[*retrieval.Result]{
		Records: []ingest.Record[*retrieval.Result]{{Payload: missing}, {Payload: matching}},
	})
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if report.Written != 2 {
		t.Fatalf("Written = %d, want 2", report.Written)
	}
	if missing.IndexIdentity != "" {
		t.Errorf("caller missing identity = %q, want unchanged", missing.IndexIdentity)
	}
	if matching.IndexIdentity != "test-space" {
		t.Errorf("caller matching identity = %q, want unchanged", matching.IndexIdentity)
	}
	if got := fi.upserts[0][0]; got == missing || got.IndexIdentity != "test-space" {
		t.Errorf("stamped item = %#v, want copied item with identity test-space", got)
	}
	if got := fi.upserts[0][1]; got == matching || got.IndexIdentity != "test-space" {
		t.Errorf("matching item = %#v, want copied item with identity test-space", got)
	}
}

func TestSinkWrite_RejectsIdentityConflictBeforeUpsert(t *testing.T) {
	t.Parallel()
	fi := &fakeIndex{}
	sink, err := NewSink(fi, Config{IndexIdentity: "test-space"})
	if err != nil {
		t.Fatalf("NewSink: %v", err)
	}
	first := &retrieval.Result{ID: "first"}
	conflict := &retrieval.Result{ID: "conflict", IndexIdentity: "other-space"}

	report, err := sink.Write(context.Background(), ingest.Batch[*retrieval.Result]{
		Records: []ingest.Record[*retrieval.Result]{{Payload: first}, {Payload: conflict}},
	})
	if !errors.Is(err, index.ErrIdentityMismatch) {
		t.Fatalf("err = %v, want errors.Is(_, %v)", err, index.ErrIdentityMismatch)
	}
	if report.Errors != 1 || report.Written != 0 {
		t.Errorf("report = %+v, want one error and no writes", report)
	}
	if len(fi.upserts) != 0 {
		t.Errorf("Upsert called %d times, want 0", len(fi.upserts))
	}
	if first.IndexIdentity != "" || conflict.IndexIdentity != "other-space" {
		t.Errorf("caller payloads mutated: first=%q conflict=%q", first.IndexIdentity, conflict.IndexIdentity)
	}
}

func TestSinkSharesMemoryIndexWithApp(t *testing.T) {
	t.Parallel()
	const identity = "hashing-32|external-chunks-v1"
	ctx := context.Background()
	idx := reliquary.NewMemoryIndex()
	embedder := embed.NewHashing(32)
	app, err := reliquary.New(
		reliquary.WithEmbedder(embedder),
		reliquary.WithIndex(idx),
		reliquary.WithIndexIdentity(identity),
	)
	if err != nil {
		t.Fatalf("reliquary.New: %v", err)
	}
	sink, err := NewSink(idx, Config{IndexIdentity: identity})
	if err != nil {
		t.Fatalf("NewSink: %v", err)
	}
	vector := embed.HashVector("shared sink result", 32)
	embedding := make([]float64, len(vector))
	for i, value := range vector {
		embedding[i] = float64(value)
	}
	_, err = sink.Write(ctx, ingest.Batch[*retrieval.Result]{Records: []ingest.Record[*retrieval.Result]{
		{Payload: &retrieval.Result{ID: "shared", Content: "shared sink result", Embedding: embedding}},
	}})
	if err != nil {
		t.Fatalf("Write: %v", err)
	}

	results, err := app.Search(ctx, "shared sink result")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 || results[0].ID != "shared" || results[0].IndexIdentity != identity {
		t.Fatalf("results = %#v, want shared result in %q", results, identity)
	}
}

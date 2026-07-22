package ingest

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestBatchCarriesCursor(t *testing.T) {
	t.Parallel()

	batch := Batch[string]{
		Records: []Record[string]{{ID: "1", Payload: "body"}},
		Cursor:  Cursor{Source: "feed", Token: "next"},
	}
	if batch.Cursor.Token != "next" || len(batch.Records) != 1 {
		t.Fatalf("unexpected batch: %#v", batch)
	}
}

func TestPipelineRun(t *testing.T) {
	t.Parallel()

	sink := &collectSink[note]{}
	pipeline := NewPipeline[string, note](
		&sliceReader{lines: []string{"1|Alpha|one", "bad", "map-error|invalid", "2|Beta|two"}},
		lineDecoder{},
		noteMapper{},
		sink,
	)

	report, err := pipeline.Run(context.Background(), Cursor{Source: "lines"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if report.Read != 4 || report.Written != 2 || report.Errors != 2 || report.Cursor.Token != "" {
		t.Fatalf("Run() report = %+v, want read=4 written=2 errors=2 final cursor", report)
	}
	if len(sink.records) != 2 || sink.records[0].Payload.Title != "Alpha" || sink.records[1].Payload.Body != "two" {
		t.Fatalf("sink records = %#v", sink.records)
	}
}

func TestNewPipelineKeepsPayloadOnlyDecoderBehavior(t *testing.T) {
	t.Parallel()

	sink := &collectSink[note]{}
	pipeline := NewPipeline[string, note](
		&oneBatchReader{batch: Batch[[]byte]{
			Records: []Record[[]byte]{{
				ID:       "raw-id",
				Payload:  []byte("decoded-id|Title|body"),
				Metadata: map[string]string{"source": "raw"},
			}},
		}},
		lineDecoder{},
		noteMapper{},
		sink,
	)

	_, err := pipeline.Run(context.Background(), Cursor{})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(sink.records) != 1 || sink.records[0].ID != "decoded-id" || sink.records[0].Metadata != nil {
		t.Fatalf("sink records = %#v, want decoder-owned envelope", sink.records)
	}
}

func TestNewRecordPipelinePassesRawRecordEnvelope(t *testing.T) {
	t.Parallel()

	raw := Record[[]byte]{
		ID:       "docs/readme.md",
		Payload:  []byte("Title|body"),
		Metadata: map[string]string{"source": "docs", "filename": "readme.md"},
	}
	decoder := &envelopeDecoder{}
	sink := &collectSink[note]{}
	pipeline := NewRecordPipeline[string, note](
		&oneBatchReader{batch: Batch[[]byte]{Records: []Record[[]byte]{raw}}},
		decoder,
		noteMapper{},
		sink,
	)

	_, err := pipeline.Run(context.Background(), Cursor{})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if decoder.got.ID != raw.ID || string(decoder.got.Payload) != string(raw.Payload) || decoder.got.Metadata["filename"] != "readme.md" {
		t.Fatalf("DecodeRecord() record = %#v, want complete raw envelope", decoder.got)
	}
	if len(sink.records) != 1 || sink.records[0].ID != raw.ID || sink.records[0].Metadata["source"] != "docs" {
		t.Fatalf("sink records = %#v, want record decoder propagation", sink.records)
	}
}

func TestPipelineRunMergesSinkReportBeforeWriteErrorWithoutAdvancingCursor(t *testing.T) {
	t.Parallel()

	wantErr := fmt.Errorf("sink failed")
	start := Cursor{Source: "feed", Token: "before"}
	pipeline := NewPipeline[string, note](
		&oneBatchReader{batch: Batch[[]byte]{
			Records: []Record[[]byte]{{ID: "1", Payload: []byte("1|Alpha|one")}},
			Cursor:  Cursor{Source: "feed", Token: "failed"},
		}},
		lineDecoder{},
		noteMapper{},
		&reportingErrorSink[note]{report: Report{Read: 999, Written: 1, Skipped: 2, Errors: 3, Cursor: Cursor{Source: "feed", Token: "failed"}}, err: wantErr},
	)

	report, err := pipeline.Run(context.Background(), start)
	if err != wantErr {
		t.Fatalf("Run() error = %v, want %v", err, wantErr)
	}
	if report.Read != 1 || report.Written != 1 || report.Skipped != 2 || report.Errors != 3 {
		t.Fatalf("Run() report = %+v, want raw read and merged sink counts", report)
	}
	if report.Cursor != start {
		t.Fatalf("Run() cursor = %+v, want last successful cursor %+v", report.Cursor, start)
	}
}

func TestPipelineRunWritesEmptyContinuationAndTerminalBatches(t *testing.T) {
	t.Parallel()

	sink := &batchSink[note]{}
	pipeline := NewPipeline[string, note](
		&scriptedReader{batches: []Batch[[]byte]{
			{Cursor: Cursor{Source: "feed", Token: "after-empty"}},
			{
				Records: []Record[[]byte]{{ID: "1", Payload: []byte("1|Alpha|one")}},
				Cursor:  Cursor{Source: "feed"},
			},
		}},
		lineDecoder{},
		noteMapper{},
		sink,
	)

	report, err := pipeline.Run(context.Background(), Cursor{Source: "feed", Token: "start"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if report.Read != 1 || report.Written != 1 || report.Cursor != (Cursor{Source: "feed"}) {
		t.Fatalf("Run() report = %+v, want one record and committed terminal cursor", report)
	}
	if len(sink.batches) != 2 || len(sink.batches[0].Records) != 0 || len(sink.batches[1].Records) != 1 {
		t.Fatalf("sink batches = %#v, want empty continuation then populated terminal", sink.batches)
	}
}

func TestPipelineRunWritesEmptyTerminalBatch(t *testing.T) {
	t.Parallel()

	sink := &batchSink[note]{}
	pipeline := NewPipeline[string, note](
		&scriptedReader{batches: []Batch[[]byte]{{Cursor: Cursor{Source: "feed"}}}},
		lineDecoder{},
		noteMapper{},
		sink,
	)

	report, err := pipeline.Run(context.Background(), Cursor{Source: "feed", Token: "last"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(sink.batches) != 1 || len(sink.batches[0].Records) != 0 {
		t.Fatalf("sink batches = %#v, want one empty terminal batch", sink.batches)
	}
	if report.Cursor != (Cursor{Source: "feed"}) {
		t.Fatalf("Run() cursor = %+v, want committed terminal cursor", report.Cursor)
	}
}

func TestPipelineRunEmptyBatchSinkErrorPreservesCursor(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("checkpoint failed")
	start := Cursor{Source: "feed", Token: "before"}
	pipeline := NewPipeline[string, note](
		&scriptedReader{batches: []Batch[[]byte]{{Cursor: Cursor{Source: "feed", Token: "after"}}}},
		lineDecoder{},
		noteMapper{},
		&reportingErrorSink[note]{err: wantErr},
	)

	report, err := pipeline.Run(context.Background(), start)
	if !errors.Is(err, wantErr) {
		t.Fatalf("Run() error = %v, want %v", err, wantErr)
	}
	if report.Cursor != start {
		t.Fatalf("Run() cursor = %+v, want last committed cursor %+v", report.Cursor, start)
	}
}

func TestPipelineRunRejectsUnchangedNonterminalCursor(t *testing.T) {
	t.Parallel()

	for _, records := range [][]Record[[]byte]{
		nil,
		{{ID: "bad", Payload: []byte("bad")}},
	} {
		records := records
		name := "without records"
		if len(records) > 0 {
			name = "with records"
		}
		t.Run(name, func(t *testing.T) {
			start := Cursor{Source: "feed", Token: "same"}
			sink := &batchSink[note]{}
			pipeline := NewPipeline[string, note](
				&scriptedReader{batches: []Batch[[]byte]{{Records: records, Cursor: start}}},
				lineDecoder{},
				noteMapper{},
				sink,
			)

			report, err := pipeline.Run(context.Background(), start)
			if !errors.Is(err, ErrCursorNotAdvanced) {
				t.Fatalf("Run() error = %v, want ErrCursorNotAdvanced", err)
			}
			if report.Read != len(records) || report.Errors != 0 || report.Cursor != start {
				t.Fatalf("Run() report = %+v, want read=%d and unchanged committed cursor", report, len(records))
			}
			if len(sink.batches) != 0 {
				t.Fatalf("sink batches = %#v, want no write", sink.batches)
			}
		})
	}
}

type note struct {
	Title string
	Body  string
}

type sliceReader struct {
	lines []string
	pos   int
}

func (r *sliceReader) Read(_ context.Context, _ Cursor) (Batch[[]byte], error) {
	if r.pos >= len(r.lines) {
		return Batch[[]byte]{Cursor: Cursor{Source: "lines"}}, nil
	}
	line := r.lines[r.pos]
	r.pos++
	token := fmt.Sprintf("%d", r.pos)
	if r.pos == len(r.lines) {
		token = ""
	}
	return Batch[[]byte]{
		Records: []Record[[]byte]{{ID: line, Payload: []byte(line)}},
		Cursor:  Cursor{Source: "lines", Token: token},
	}, nil
}

type lineDecoder struct{}

func (lineDecoder) Decode(_ context.Context, data []byte) ([]Record[string], error) {
	parts := strings.SplitN(string(data), "|", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("malformed line: %q", data)
	}
	return []Record[string]{{ID: parts[0], Payload: parts[1]}}, nil
}

type envelopeDecoder struct {
	got Record[[]byte]
}

func (d *envelopeDecoder) DecodeRecord(_ context.Context, record Record[[]byte]) ([]Record[string], error) {
	d.got = record
	return []Record[string]{{ID: record.ID, Payload: string(record.Payload), Metadata: record.Metadata}}, nil
}

type noteMapper struct{}

func (noteMapper) Map(_ context.Context, rec Record[string]) (Record[note], error) {
	parts := strings.SplitN(rec.Payload, "|", 2)
	if len(parts) != 2 {
		return Record[note]{}, fmt.Errorf("malformed payload: %q", rec.Payload)
	}
	return Record[note]{ID: rec.ID, Payload: note{Title: parts[0], Body: parts[1]}, Metadata: rec.Metadata}, nil
}

type collectSink[T any] struct {
	records []Record[T]
}

type oneBatchReader struct {
	batch Batch[[]byte]
	read  bool
}

type scriptedReader struct {
	batches []Batch[[]byte]
	pos     int
}

func (r *scriptedReader) Read(_ context.Context, _ Cursor) (Batch[[]byte], error) {
	batch := r.batches[r.pos]
	r.pos++
	return batch, nil
}

func (r *oneBatchReader) Read(_ context.Context, _ Cursor) (Batch[[]byte], error) {
	if r.read {
		return Batch[[]byte]{}, nil
	}
	r.read = true
	return r.batch, nil
}

type reportingErrorSink[T any] struct {
	report Report
	err    error
}

func (s *reportingErrorSink[T]) Write(_ context.Context, _ Batch[T]) (Report, error) {
	return s.report, s.err
}

func (s *collectSink[T]) Write(_ context.Context, batch Batch[T]) (Report, error) {
	s.records = append(s.records, batch.Records...)
	return Report{Read: len(batch.Records), Written: len(batch.Records), Cursor: batch.Cursor}, nil
}

type batchSink[T any] struct {
	batches []Batch[T]
}

func (s *batchSink[T]) Write(_ context.Context, batch Batch[T]) (Report, error) {
	s.batches = append(s.batches, batch)
	return Report{Written: len(batch.Records), Cursor: batch.Cursor}, nil
}

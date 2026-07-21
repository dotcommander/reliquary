package ingest

import (
	"context"
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
		&sliceReader{lines: []string{"1|Alpha|one", "bad", "2|Beta|two"}},
		lineDecoder{},
		noteMapper{},
		sink,
	)

	report, err := pipeline.Run(context.Background(), Cursor{Source: "lines"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if report.Read != 3 || report.Written != 2 || report.Errors != 1 || report.Cursor.Token != "" {
		t.Fatalf("Run() report = %+v, want read=3 written=2 errors=1 final cursor", report)
	}
	if len(sink.records) != 2 || sink.records[0].Payload.Title != "Alpha" || sink.records[1].Payload.Body != "two" {
		t.Fatalf("sink records = %#v", sink.records)
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

type noteMapper struct{}

func (noteMapper) Map(_ context.Context, rec Record[string]) (Record[note], error) {
	parts := strings.SplitN(rec.Payload, "|", 2)
	if len(parts) != 2 {
		return Record[note]{}, fmt.Errorf("malformed payload: %q", rec.Payload)
	}
	return Record[note]{ID: rec.ID, Payload: note{Title: parts[0], Body: parts[1]}}, nil
}

type collectSink[T any] struct {
	records []Record[T]
}

type oneBatchReader struct {
	batch Batch[[]byte]
	read  bool
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

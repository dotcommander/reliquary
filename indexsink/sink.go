// Package indexsink adapts pipeline/ingest to reliquary's Index. It is the
// termination glue that lets a resumable batch pipeline land in the same
// storage that App.Ingest uses (Index.Upsert), without forcing reliquary's
// chunk+embed transform into the generic pipeline contracts.
package indexsink

import (
	"context"

	"github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/pipeline/ingest"
	"github.com/dotcommander/reliquary/retrieval"
)

// Sink persists pipeline records into a reliquary Index.
type Sink struct {
	idx index.Index
}

// NewSink returns an ingest.Sink[*retrieval.Result] backed by idx. Pass the same
// Index wired into App via WithIndex.
func NewSink(idx index.Index) ingest.Sink[*retrieval.Result] {
	return &Sink{idx: idx}
}

// Write upserts every record's payload into the Index. Nil payloads are skipped.
// On Upsert error the run aborts: pipeline/ingest treats write errors as fatal
// because persistence state would be ambiguous (see Pipeline.Run).
func (s *Sink) Write(ctx context.Context, batch ingest.Batch[*retrieval.Result]) (ingest.Report, error) {
	items := make([]*retrieval.Result, 0, len(batch.Records))
	skipped := 0
	for _, r := range batch.Records {
		if r.Payload == nil {
			skipped++
			continue
		}
		items = append(items, r.Payload)
	}
	report := ingest.Report{Skipped: skipped, Cursor: batch.Cursor}
	if len(items) == 0 {
		return report, nil
	}
	if err := s.idx.Upsert(ctx, items); err != nil {
		report.Errors = len(items)
		return report, err
	}
	report.Written = len(items)
	return report, nil
}

// Package indexsink adapts pipeline/ingest to reliquary's Index. It is the
// termination glue that lets a resumable batch pipeline share the Index wired
// into an App without forcing reliquary's chunk+embed transform into the
// generic pipeline contracts. The sink uses merge-style Index.Upsert;
// App.Ingest atomically replaces complete documents with Index.ReplaceDocuments.
package indexsink

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/internal/validate"
	"github.com/dotcommander/reliquary/pipeline/ingest"
	"github.com/dotcommander/reliquary/retrieval"
)

var (
	// ErrNilIndex reports a nil or typed-nil Index passed to NewSink.
	ErrNilIndex = errors.New("indexsink: index must not be nil")
	// ErrInvalidIndexIdentity reports a blank index identity passed to NewSink.
	ErrInvalidIndexIdentity = errors.New("indexsink: index identity must not be blank")
)

// Config identifies the embedding and chunking space written by a Sink.
type Config struct {
	IndexIdentity string
}

// Sink persists pipeline records into a reliquary Index.
type Sink struct {
	idx           index.Index
	indexIdentity string
}

// NewSink returns a Sink backed by idx. Pass the same Index and index identity
// wired into App via WithIndex and WithIndexIdentity.
func NewSink(idx index.Index, config Config) (*Sink, error) {
	if validate.IsNil(idx) {
		return nil, ErrNilIndex
	}
	if strings.TrimSpace(config.IndexIdentity) == "" {
		return nil, ErrInvalidIndexIdentity
	}
	return &Sink{idx: idx, indexIdentity: config.IndexIdentity}, nil
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
		item := *r.Payload
		switch item.IndexIdentity {
		case "":
			item.IndexIdentity = s.indexIdentity
		case s.indexIdentity:
		default:
			report := ingest.Report{Read: len(batch.Records), Skipped: skipped, Errors: 1, Cursor: batch.Cursor}
			return report, fmt.Errorf("%w: sink has %q, item %q has %q", index.ErrIdentityMismatch, s.indexIdentity, item.ID, item.IndexIdentity)
		}
		items = append(items, &item)
	}
	report := ingest.Report{Read: len(batch.Records), Skipped: skipped, Cursor: batch.Cursor}
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

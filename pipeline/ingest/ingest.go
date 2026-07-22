// Package ingest defines generic ingestion contracts without source policy.
package ingest

import (
	"context"
	"errors"
)

// ErrCursorNotAdvanced reports a nonterminal page that repeats the requested
// cursor. Continuing would reread the same page indefinitely.
var ErrCursorNotAdvanced = errors.New("ingest: cursor not advanced")

// Cursor identifies a resumable position in a caller-owned source.
type Cursor struct {
	Source string
	Token  string
}

// Record is the generic unit passed through an ingest pipeline.
type Record[T any] struct {
	ID       string
	Payload  T
	Metadata map[string]string
}

// Batch groups records with their resume cursor.
type Batch[T any] struct {
	Records []Record[T]
	Cursor  Cursor
}

// Report summarizes an ingest run without owning logging or persistence.
type Report struct {
	Read    int
	Written int
	Skipped int
	Errors  int
	Cursor  Cursor
}

// Reader reads batches from a source.
type Reader[T any] interface {
	Read(ctx context.Context, cursor Cursor) (Batch[T], error)
}

// Decoder converts raw source bytes into records.
type Decoder[T any] interface {
	Decode(ctx context.Context, data []byte) ([]Record[T], error)
}

// RecordDecoder converts a raw record while retaining its ID and metadata.
// Implementations explicitly own propagation of the input record envelope.
type RecordDecoder[T any] interface {
	DecodeRecord(ctx context.Context, record Record[[]byte]) ([]Record[T], error)
}

// Mapper maps records between semantic spaces.
type Mapper[In, Out any] interface {
	Map(ctx context.Context, record Record[In]) (Record[Out], error)
}

// Sink writes records to a caller-owned destination.
type Sink[T any] interface {
	Write(ctx context.Context, batch Batch[T]) (Report, error)
}

// Package index defines the candidate-retrieval boundary used by Reliquary.
package index

import (
	"context"
	"errors"

	"github.com/dotcommander/reliquary/retrieval"
)

// ErrDimensionMismatch reports incompatible query and indexed vector spaces.
var ErrDimensionMismatch = errors.New("reliquary index: vector dimension mismatch")

// ErrIdentityMismatch reports an attempt to mix different embedding or
// chunking spaces in one index.
var ErrIdentityMismatch = errors.New("reliquary index: identity mismatch")

// Index stores embedded chunks and retrieves a bounded candidate set.
type Index interface {
	Upsert(ctx context.Context, items []*retrieval.Result) error
	DeleteDocument(ctx context.Context, documentID string) error
	Search(ctx context.Context, query IndexQuery) ([]*retrieval.Result, error)
}

// Resetter is implemented by indexes that can destructively remove every
// stored result. It is optional so custom Index implementations remain source
// compatible.
type Resetter interface {
	Reset(ctx context.Context) error
}

// IndexQuery describes candidate retrieval. A zero Limit leaves the candidate
// count implementation-defined or unbounded.
type IndexQuery struct {
	Identity string
	Text     string
	Vector   []float32
	Limit    int
	Filter   map[string]any
}

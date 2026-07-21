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

// ErrInvalidDocumentID reports a blank document identifier.
var ErrInvalidDocumentID = errors.New("reliquary index: document ID must not be blank")

// ErrDuplicateDocumentID reports a replacement batch that names one document
// more than once.
var ErrDuplicateDocumentID = errors.New("reliquary index: duplicate document ID")

// ErrResultIDConflict reports a replacement result ID that is duplicated in
// the batch or already belongs to a retained document.
var ErrResultIDConflict = errors.New("reliquary index: replacement result ID conflict")

// DocumentReplacement atomically replaces every indexed result whose
// Result.DocumentID equals DocumentID. Result ID naming does not imply
// ownership. An empty Results slice deletes the document.
type DocumentReplacement struct {
	DocumentID string
	Results    []*retrieval.Result
}

// Index stores embedded chunks and retrieves a bounded candidate set. Non-empty
// result embeddings and query vectors must contain only finite values; empty
// embeddings remain valid for lexical-only results. The first non-nil result
// establishes its identity and the first embedded result establishes its vector
// dimension. Implementations must preserve that space across deletes and
// replacements until Resetter.Reset succeeds.
type Index interface {
	Upsert(ctx context.Context, items []*retrieval.Result) error
	// ReplaceDocuments applies the complete slice atomically: either every
	// document revision is replaced or none is. Empty Results deletes that
	// document. Result IDs must be unique within the batch and must not collide
	// with retained documents; implementations return ErrResultIDConflict when
	// either invariant is violated. Replacement never changes the established
	// index identity or vector dimension; Reset is the only escape hatch.
	ReplaceDocuments(ctx context.Context, replacements []DocumentReplacement) error
	// DeleteDocument removes only results whose Result.DocumentID exactly equals
	// documentID. Result ID naming does not imply ownership. It rejects empty or
	// whitespace-only IDs with ErrInvalidDocumentID and leaves the index unchanged.
	DeleteDocument(ctx context.Context, documentID string) error
	Search(ctx context.Context, query IndexQuery) ([]*retrieval.Result, error)
}

// Resetter is implemented by indexes that can destructively remove every
// stored result. It is optional so custom Index implementations remain source
// compatible.
type Resetter interface {
	Reset(ctx context.Context) error
}

// IndexQuery describes candidate retrieval. Filter values are JSON scalars.
// Metadata keys must be present to match; explicit nil matches only JSON null.
// Strings and booleans are type-exact, while finite JSON numbers compare by
// exact numeric value across accepted Go numeric types and json.Number. The
// reserved id, document_id, and filename fields match strings only. A zero
// Limit leaves the candidate count implementation-defined or unbounded.
type IndexQuery struct {
	Identity string
	Text     string
	Vector   []float32
	Limit    int
	Filter   map[string]any
}

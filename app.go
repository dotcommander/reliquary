// Package reliquary is the high-level facade for the reliquary retrieval
// toolkit. It wires the chunking, embedding, and retrieval pipelines behind a
// small App value with sensible defaults, so a caller can ingest documents and
// run hybrid search without assembling each lower-level package by hand.
//
// Construct an App with New (bring your own embedder) or with the zero-config
// Quickstart/InMemory conventions. The App owns no resources and starts
// nothing; every call takes an explicit context.
package reliquary

import (
	"context"
	"errors"
	"strings"

	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/embedding"
	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/index/inmem"
	"github.com/dotcommander/reliquary/internal/validate"
	"github.com/dotcommander/reliquary/retrieval"
)

// ErrNoEmbedder is returned by New when no embedder is supplied. There is no
// sensible default embedding model, so the embedder is the one required
// dependency.
var ErrNoEmbedder = errors.New("reliquary: an embedder is required (use WithEmbedder)")

// ErrNilApp is returned when a method is called on a nil App.
var ErrNilApp = errors.New("reliquary: nil app")

// ErrInvalidIndexIdentity is returned when WithIndexIdentity is empty.
var ErrInvalidIndexIdentity = errors.New("reliquary: index identity must not be empty")

// ErrResetUnsupported reports an Index without the optional reset contract.
var ErrResetUnsupported = errors.New("reliquary: index does not support reset")

// ErrIdentityMismatch reports incompatible index identities.
var ErrIdentityMismatch = indexcontract.ErrIdentityMismatch

// ErrInvalidDocumentID reports a blank document identifier passed to Ingest.
var ErrInvalidDocumentID = indexcontract.ErrInvalidDocumentID

// ErrDuplicateDocumentID reports duplicate document identifiers in one Ingest call.
var ErrDuplicateDocumentID = indexcontract.ErrDuplicateDocumentID

// ErrResultIDConflict reports an invalid replacement result-ID collision.
var ErrResultIDConflict = indexcontract.ErrResultIDConflict

// App is a wired retrieval pipeline. Construct it with New, Quickstart, or
// InMemory.
type App struct {
	embedder         embeddings.Embedder
	index            indexcontract.Index
	strategy         chunking.Strategy
	size             int
	overlap          int
	weights          retrieval.Weights
	indexIdentity    string
	indexIdentitySet bool
}

// Option configures an App. Every default is explicit and overridable.
type Option func(*App)

// New builds an App from the supplied options. An embedder and a non-empty
// index identity are required; all other settings default: an in-memory index,
// smart-boundary chunking at 220 characters with no overlap, and the default
// hybrid-scoring weights.
func New(opts ...Option) (*App, error) {
	a := &App{
		index:    inmem.New(),
		strategy: chunking.SmartBoundary,
		size:     220,
		overlap:  0,
		weights:  retrieval.DefaultWeights(),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(a)
		}
	}
	if validate.IsNil(a.embedder) {
		return nil, ErrNoEmbedder
	}
	if !a.indexIdentitySet || strings.TrimSpace(a.indexIdentity) == "" {
		return nil, ErrInvalidIndexIdentity
	}
	if validate.IsNil(a.index) {
		a.index = inmem.New()
	}
	return a, nil
}

// ensureReady is the shared precondition for every App operation: a non-nil
// App with the required embedder and a non-nil index. Calling it on a nil *App
// is safe.
func (a *App) ensureReady() error {
	if a == nil {
		return ErrNilApp
	}
	if validate.IsNil(a.embedder) {
		return ErrNoEmbedder
	}
	if validate.IsNil(a.index) {
		a.index = inmem.New()
	}
	return nil
}

// WithEmbedder sets the embedder used for ingestion and queries (required).
func WithEmbedder(e embeddings.Embedder) Option {
	return func(a *App) { a.embedder = e }
}

// Index is the candidate-retrieval boundary used by App.
type Index = indexcontract.Index

// IndexQuery describes candidate retrieval.
type IndexQuery = indexcontract.IndexQuery

// DocumentReplacement describes one complete document revision for an Index.
type DocumentReplacement = indexcontract.DocumentReplacement

// WithIndex sets the index used for ingestion and candidate retrieval. A nil
// or typed-nil Index is unset and falls back to the default in-memory index.
func WithIndex(index Index) Option {
	return func(a *App) {
		a.index = index
	}
}

// WithIndexIdentity identifies the embedding vector space and chunking policy
// used by this App. Index implementations reject reads and writes against a
// different identity, even when vector dimensions match.
func WithIndexIdentity(identity string) Option {
	return func(a *App) {
		a.indexIdentity = identity
		a.indexIdentitySet = true
	}
}

// NewMemoryIndex returns the default concurrency-safe in-process Index.
func NewMemoryIndex() Index { return inmem.New() }

// ResetIndex destructively removes every indexed chunk. It is intended as the
// explicit rebuild escape hatch after changing index identity; source
// documents must be ingested again. Custom indexes may opt in via index.Resetter.
func (a *App) ResetIndex(ctx context.Context) error {
	if err := a.ensureReady(); err != nil {
		return err
	}
	resetter, ok := a.index.(indexcontract.Resetter)
	if !ok {
		return ErrResetUnsupported
	}
	return resetter.Reset(ctx)
}

// WithChunker overrides the chunking strategy and chunk sizing.
func WithChunker(strategy chunking.Strategy, size, overlap int) Option {
	return func(a *App) {
		a.strategy = strategy
		a.size = size
		a.overlap = overlap
	}
}

// WithWeights overrides the default hybrid-scoring weights.
func WithWeights(w retrieval.Weights) Option {
	return func(a *App) { a.weights = w }
}

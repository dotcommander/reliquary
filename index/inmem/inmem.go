// Package inmem provides Reliquary's concurrency-safe in-process index.
package inmem

import (
	"context"
	"fmt"
	"strings"
	"sync"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/internal/indexutil"
	"github.com/dotcommander/reliquary/retrieval"
)

// Index is an in-memory index keyed by retrieval result ID.
type Index struct {
	mu        sync.RWMutex
	items     map[string]*retrieval.Result
	dimension int
	identity  string
}

// New returns an empty in-memory index.
func New() *Index { return &Index{items: make(map[string]*retrieval.Result)} }

// Reset destructively removes all indexed results and their identity.
func (i *Index) Reset(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	i.items = make(map[string]*retrieval.Result)
	i.dimension = 0
	i.identity = ""
	return nil
}

// Upsert inserts new items and replaces existing items with the same ID.
func (i *Index) Upsert(ctx context.Context, items []*retrieval.Result) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	dimension := i.dimension
	identity := i.identity
	identityEstablished := len(i.items) > 0
	for _, item := range items {
		if err := ctx.Err(); err != nil {
			return err
		}
		if item == nil {
			continue
		}
		if item.ID == "" {
			return fmt.Errorf("reliquary index: empty item ID")
		}
		if !identityEstablished {
			identity = item.IndexIdentity
			identityEstablished = true
		} else if item.IndexIdentity != identity {
			return fmt.Errorf("%w: index has %q, item %q has %q", indexcontract.ErrIdentityMismatch, identity, item.ID, item.IndexIdentity)
		}
		if len(item.Embedding) > 0 {
			if dimension == 0 {
				dimension = len(item.Embedding)
			} else if len(item.Embedding) != dimension {
				return fmt.Errorf("%w: index has %d dimensions, item %q has %d", indexcontract.ErrDimensionMismatch, dimension, item.ID, len(item.Embedding))
			}
		}
	}
	if i.items == nil {
		i.items = make(map[string]*retrieval.Result)
	}
	for _, item := range items {
		if item != nil {
			i.items[item.ID] = indexutil.Clone(item)
		}
	}
	i.dimension = dimension
	i.identity = identity
	return nil
}

// DeleteDocument removes all chunks belonging to documentID.
func (i *Index) DeleteDocument(ctx context.Context, documentID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	for id, item := range i.items {
		if item.DocumentID == documentID || item.DocumentID == "" && strings.HasPrefix(id, documentID+"#") {
			delete(i.items, id)
		}
	}
	i.dimension = 0
	i.identity = ""
	for _, item := range i.items {
		i.identity = item.IndexIdentity
		if len(item.Embedding) > 0 {
			i.dimension = len(item.Embedding)
			break
		}
	}
	return nil
}

// Search scans the in-memory entries and returns scored candidates.
func (i *Index) Search(ctx context.Context, query indexcontract.IndexQuery) ([]*retrieval.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	i.mu.RLock()
	items := make([]*retrieval.Result, 0, len(i.items))
	for _, item := range i.items {
		if err := ctx.Err(); err != nil {
			i.mu.RUnlock()
			return nil, err
		}
		items = append(items, indexutil.Clone(item))
	}
	dimension := i.dimension
	identity := i.identity
	identityEstablished := len(i.items) > 0
	i.mu.RUnlock()
	if identityEstablished && identity != query.Identity {
		return nil, fmt.Errorf("%w: index has %q, query has %q", indexcontract.ErrIdentityMismatch, identity, query.Identity)
	}
	if dimension > 0 && len(query.Vector) > 0 && len(query.Vector) != dimension {
		return nil, fmt.Errorf("%w: index has %d dimensions, query has %d", indexcontract.ErrDimensionMismatch, dimension, len(query.Vector))
	}
	return indexutil.Search(ctx, query, items)
}

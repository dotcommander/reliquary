// Package inmem provides Reliquary's concurrency-safe in-process index.
package inmem

import (
	"context"
	"fmt"
	"sync"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/internal/indexutil"
	"github.com/dotcommander/reliquary/retrieval"
)

// Index is an in-memory index keyed by retrieval result ID.
type Index struct {
	mu    sync.RWMutex
	items map[string]*retrieval.Result
	space indexutil.Space
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
	i.space = indexutil.Space{}
	return nil
}

// Upsert inserts new items and replaces existing items with the same ID.
func (i *Index) Upsert(ctx context.Context, items []*retrieval.Result) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
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
	}
	nextSpace, err := i.space.ValidateResults(items)
	if err != nil {
		return err
	}
	if i.items == nil {
		i.items = make(map[string]*retrieval.Result)
	}
	for _, item := range items {
		if item != nil {
			i.items[item.ID] = indexutil.Clone(item)
		}
	}
	i.space = nextSpace
	return nil
}

// ReplaceDocuments atomically replaces complete document revisions. Empty
// result sets delete their document.
func (i *Index) ReplaceDocuments(ctx context.Context, replacements []indexcontract.DocumentReplacement) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := indexutil.ValidateReplacements(replacements); err != nil {
		return err
	}
	seen := make(map[string]struct{}, len(replacements))
	for _, replacement := range replacements {
		seen[replacement.DocumentID] = struct{}{}
	}

	i.mu.Lock()
	defer i.mu.Unlock()
	nextSpace := i.space
	for _, replacement := range replacements {
		var err error
		nextSpace, err = nextSpace.ValidateResults(replacement.Results)
		if err != nil {
			return err
		}
	}
	next := make(map[string]*retrieval.Result, len(i.items))
	for id, item := range i.items {
		if err := ctx.Err(); err != nil {
			return err
		}
		remove := false
		for documentID := range seen {
			if item.DocumentID == documentID {
				remove = true
				break
			}
		}
		if !remove {
			next[id] = indexutil.Clone(item)
		}
	}
	for _, replacement := range replacements {
		for _, item := range replacement.Results {
			if err := ctx.Err(); err != nil {
				return err
			}
			if item != nil {
				if retained := next[item.ID]; retained != nil {
					return fmt.Errorf("%w: %q belongs to retained document %q", indexcontract.ErrResultIDConflict, item.ID, retained.DocumentID)
				}
				next[item.ID] = indexutil.Clone(item)
			}
		}
	}

	i.items = next
	i.space = nextSpace
	return nil
}

// DeleteDocument removes all chunks belonging to documentID.
func (i *Index) DeleteDocument(ctx context.Context, documentID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := indexutil.ValidateDocumentID(documentID); err != nil {
		return err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	for id, item := range i.items {
		if item.DocumentID == documentID {
			delete(i.items, id)
		}
	}
	return nil
}

// Search scans the in-memory entries and returns scored candidates.
func (i *Index) Search(ctx context.Context, query indexcontract.IndexQuery) ([]*retrieval.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := indexutil.ValidateFilter(query.Filter); err != nil {
		return nil, fmt.Errorf("reliquary index: %w", err)
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
	space := i.space
	i.mu.RUnlock()
	if err := space.ValidateQuery(query); err != nil {
		return nil, err
	}
	return indexutil.Search(ctx, query, items)
}

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
	mu        sync.RWMutex
	items     map[string]*retrieval.Result
	documents map[string]map[string]struct{} // document ID -> result IDs
	space     indexutil.Space
}

// New returns an empty in-memory index.
func New() *Index {
	return &Index{
		items:     make(map[string]*retrieval.Result),
		documents: make(map[string]map[string]struct{}),
	}
}

// Reset destructively removes all indexed results and their identity.
func (i *Index) Reset(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	i.items = make(map[string]*retrieval.Result)
	i.documents = make(map[string]map[string]struct{})
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
	if i.documents == nil {
		i.documents = make(map[string]map[string]struct{})
	}
	for _, item := range items {
		if item == nil {
			continue
		}
		if old := i.items[item.ID]; old != nil {
			delete(i.documents[old.DocumentID], item.ID)
			if len(i.documents[old.DocumentID]) == 0 {
				delete(i.documents, old.DocumentID)
			}
		}
		i.items[item.ID] = indexutil.Clone(item)
		if i.documents[item.DocumentID] == nil {
			i.documents[item.DocumentID] = make(map[string]struct{})
		}
		i.documents[item.DocumentID][item.ID] = struct{}{}
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
	// Stage all fallible work before mutating either map. Search takes a snapshot
	// under the same lock, so readers see only complete revisions.
	staged := make(map[string]*retrieval.Result)
	for _, replacement := range replacements {
		for _, item := range replacement.Results {
			if err := ctx.Err(); err != nil {
				return err
			}
			if item == nil {
				continue
			}
			if retained := i.items[item.ID]; retained != nil {
				if _, replaced := seen[retained.DocumentID]; !replaced {
					return fmt.Errorf("%w: %q belongs to retained document %q", indexcontract.ErrResultIDConflict, item.ID, retained.DocumentID)
				}
			}
			staged[item.ID] = indexutil.Clone(item)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if i.items == nil {
		i.items = make(map[string]*retrieval.Result)
	}
	if i.documents == nil {
		i.documents = make(map[string]map[string]struct{})
	}
	for documentID := range seen {
		for id := range i.documents[documentID] {
			delete(i.items, id)
		}
		delete(i.documents, documentID)
	}
	for id, item := range staged {
		i.items[id] = item
		if i.documents[item.DocumentID] == nil {
			i.documents[item.DocumentID] = make(map[string]struct{})
		}
		i.documents[item.DocumentID][id] = struct{}{}
	}
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
	for id := range i.documents[documentID] {
		delete(i.items, id)
	}
	delete(i.documents, documentID)
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
		items = append(items, item)
	}
	space := i.space
	i.mu.RUnlock()
	if err := space.ValidateQuery(query); err != nil {
		return nil, err
	}
	return indexutil.Search(ctx, query, items)
}

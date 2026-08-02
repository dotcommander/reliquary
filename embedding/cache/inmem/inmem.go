package inmem

import (
	"context"
	"fmt"
	"sync"

	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/embedding/cache"
)

// Store keeps embedding cache entries in process memory. It is safe for
// concurrent use, and its zero value is ready to use.
type Store struct {
	mu      sync.RWMutex
	entries map[string]cache.Entry
}

var _ cache.Store = (*Store)(nil)

// New constructs an empty Store.
func New() *Store {
	return &Store{entries: make(map[string]cache.Entry)}
}

// Get returns an isolated copy of the entry stored under key.
func (s *Store) Get(ctx context.Context, key string) (cache.Entry, bool, error) {
	if err := contextError(ctx, "get"); err != nil {
		return cache.Entry{}, false, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := contextError(ctx, "get"); err != nil {
		return cache.Entry{}, false, err
	}

	entry, ok := s.entries[key]
	if !ok {
		return cache.Entry{}, false, nil
	}
	entry.Vector = cloneVector(entry.Vector)
	return entry, true, nil
}

// Set stores an isolated copy of entry under key, replacing any prior value.
func (s *Store) Set(ctx context.Context, key string, entry cache.Entry) error {
	if err := contextError(ctx, "set"); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := contextError(ctx, "set"); err != nil {
		return err
	}

	if s.entries == nil {
		s.entries = make(map[string]cache.Entry)
	}
	entry.Vector = cloneVector(entry.Vector)
	s.entries[key] = entry
	return nil
}

func cloneVector(vector embedding.Vector) embedding.Vector {
	return append(embedding.Vector(nil), vector...)
}

func contextError(ctx context.Context, operation string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("embedding cache inmem: %s: %w", operation, err)
	}
	return nil
}

// Package indextest provides a reusable contract suite for Index implementations.
package indextest

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

// Factory returns a new, empty Index for each contract subtest.
type Factory func() indexcontract.Index

// Run exercises the behavior required of every Index implementation.
func Run(t *testing.T, newIndex Factory) {
	t.Helper()

	t.Run("upsert overwrite and latest value", func(t *testing.T) {
		idx := newIndex()
		items := []*retrieval.Result{
			{ID: "a", DocumentID: "doc-a", Filename: "a.md", Content: "old", Embedding: []float64{1, 0}, Metadata: map[string]any{"tenant": "one"}},
			{ID: "b", DocumentID: "doc-b", Filename: "b.md", Content: "other", Embedding: []float64{0, 1}},
			{ID: "a", DocumentID: "doc-a", Filename: "latest.md", Content: "latest", Embedding: []float64{1, 0}, Metadata: map[string]any{"tenant": "two"}},
		}
		mustUpsert(t, idx, items)
		got := mustSearch(t, idx, indexcontract.IndexQuery{Vector: []float32{1, 0}})
		if len(got) != 2 || got[0].ID != "a" || got[0].Content != "latest" || got[0].Filename != "latest.md" || got[0].Metadata["tenant"] != "two" {
			t.Fatalf("Search after duplicate batch = %#v", got)
		}

		mustUpsert(t, idx, []*retrieval.Result{{ID: "a", DocumentID: "doc-a", Content: "replacement", Embedding: []float64{1, 0}}})
		got = mustSearch(t, idx, indexcontract.IndexQuery{Filter: map[string]any{"id": "a"}})
		if len(got) != 1 || got[0].Content != "replacement" {
			t.Fatalf("Search after overwrite = %#v", got)
		}
	})

	t.Run("delete resets empty index dimension", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{
			{ID: "a", DocumentID: "doc-a", Embedding: []float64{1, 0}},
			{ID: "unembedded", DocumentID: "doc-b"},
		})
		if err := idx.DeleteDocument(context.Background(), "doc-a"); err != nil {
			t.Fatalf("DeleteDocument: %v", err)
		}
		mustUpsert(t, idx, []*retrieval.Result{{ID: "b", DocumentID: "doc-c", Embedding: []float64{1, 0, 0}}})
		got := mustSearch(t, idx, indexcontract.IndexQuery{Vector: []float32{1, 0, 0}})
		if len(got) != 2 || got[0].ID != "b" {
			t.Fatalf("Search after delete and dimension change = %#v", got)
		}
	})

	t.Run("limits and deterministic ordering", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{
			{ID: "z", Embedding: []float64{1, 0}},
			{ID: "a", Embedding: []float64{1, 0}},
			{ID: "middle", Embedding: []float64{0, 1}},
		})
		for _, limit := range []int{0, -3} {
			got := mustSearch(t, idx, indexcontract.IndexQuery{Vector: []float32{1, 0}, Limit: limit})
			if ids := resultIDs(got); ids != "a,z,middle" {
				t.Fatalf("limit %d order = %s, want a,z,middle", limit, ids)
			}
		}
		got := mustSearch(t, idx, indexcontract.IndexQuery{Vector: []float32{1, 0}, Limit: 2})
		if ids := resultIDs(got); ids != "a,z" {
			t.Fatalf("positive limit order = %s, want a,z", ids)
		}
	})

	t.Run("filters", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{
			{ID: "a", DocumentID: "doc-a", Filename: "a.md", Embedding: []float64{1, 0}, Metadata: map[string]any{"tenant": "one", "active": true}},
			{ID: "b", DocumentID: "doc-b", Filename: "b.md", Embedding: []float64{1, 0}, Metadata: map[string]any{"tenant": "two", "active": false}},
		})
		filters := []map[string]any{
			{"id": "a"}, {"document_id": "doc-a"}, {"filename": "a.md"},
			{"tenant": "one"}, {"tenant": "one", "active": true},
		}
		for _, filter := range filters {
			got := mustSearch(t, idx, indexcontract.IndexQuery{Filter: filter})
			if len(got) != 1 || got[0].ID != "a" {
				t.Fatalf("filter %#v = %#v", filter, got)
			}
		}
	})

	t.Run("cancellation", func(t *testing.T) {
		idx := newIndex()
		items := make([]*retrieval.Result, 32)
		for n := range items {
			items[n] = &retrieval.Result{ID: fmt.Sprintf("item-%02d", n), Embedding: []float64{1, 0}}
		}
		mustUpsert(t, idx, items)

		canceled, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := idx.Search(canceled, indexcontract.IndexQuery{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-canceled Search error = %v", err)
		}
		ctx := newCancelAfterContext(5)
		if _, err := idx.Search(ctx, indexcontract.IndexQuery{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("mid-traversal Search error = %v", err)
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{{ID: "a", Embedding: []float64{1, 0}}})
		if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "b", Embedding: []float64{1}}}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
			t.Fatalf("Upsert mismatch error = %v", err)
		}
		if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Vector: []float32{1}}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
			t.Fatalf("Search mismatch error = %v", err)
		}
	})

	t.Run("identity mismatch", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{{ID: "a", IndexIdentity: "model-a|chunks-v1", Embedding: []float64{1, 0}}})
		if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "b", IndexIdentity: "model-b|chunks-v1", Embedding: []float64{1, 0}}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
			t.Fatalf("Upsert identity mismatch error = %v", err)
		}
		if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Identity: "model-a|chunks-v2", Vector: []float32{1, 0}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
			t.Fatalf("Search identity mismatch error = %v", err)
		}
		got := mustSearch(t, idx, indexcontract.IndexQuery{Identity: "model-a|chunks-v1", Vector: []float32{1, 0}})
		if len(got) != 1 || got[0].IndexIdentity != "model-a|chunks-v1" {
			t.Fatalf("identity round trip = %#v", got)
		}
	})

	t.Run("destructive reset permits new identity", func(t *testing.T) {
		idx := newIndex()
		resetter, ok := idx.(indexcontract.Resetter)
		if !ok {
			t.Skip("index does not implement optional reset contract")
		}
		mustUpsert(t, idx, []*retrieval.Result{{ID: "old", IndexIdentity: "old", Embedding: []float64{1, 0}}})
		if err := resetter.Reset(context.Background()); err != nil {
			t.Fatalf("Reset: %v", err)
		}
		mustUpsert(t, idx, []*retrieval.Result{{ID: "new", IndexIdentity: "new", Embedding: []float64{1, 0}}})
		got := mustSearch(t, idx, indexcontract.IndexQuery{Identity: "new", Vector: []float32{1, 0}})
		if len(got) != 1 || got[0].ID != "new" {
			t.Fatalf("Search after Reset = %#v", got)
		}
	})

	t.Run("mutation isolation and round trip", func(t *testing.T) {
		idx := newIndex()
		item := &retrieval.Result{ID: "stable", DocumentID: "doc", Filename: "doc.md", Content: "original", Embedding: []float64{1, 0}, Metadata: map[string]any{"tenant": "one"}}
		mustUpsert(t, idx, []*retrieval.Result{item})
		item.Content = "input mutation"
		item.Embedding[0] = 0
		item.Metadata["tenant"] = "changed"

		got := mustSearch(t, idx, indexcontract.IndexQuery{Filter: map[string]any{"id": "stable"}})
		if len(got) != 1 || got[0].ID != "stable" || got[0].DocumentID != "doc" || got[0].Filename != "doc.md" || got[0].Content != "original" || got[0].Embedding[0] != 1 || got[0].Metadata["tenant"] != "one" {
			t.Fatalf("round trip after input mutation = %#v", got)
		}
		got[0].Embedding[0] = 0
		got[0].Metadata["tenant"] = "returned mutation"
		again := mustSearch(t, idx, indexcontract.IndexQuery{Filter: map[string]any{"id": "stable"}})
		if again[0].Embedding[0] != 1 || again[0].Metadata["tenant"] != "one" {
			t.Fatalf("stored state aliased returned result: %#v", again[0])
		}
	})

	t.Run("empty index", func(t *testing.T) {
		got := mustSearch(t, newIndex(), indexcontract.IndexQuery{})
		if len(got) != 0 {
			t.Fatalf("empty Search = %#v", got)
		}
		got = mustSearch(t, newIndex(), indexcontract.IndexQuery{Identity: "new-space"})
		if len(got) != 0 {
			t.Fatalf("identified empty Search = %#v", got)
		}
	})
}

type cancelAfterContext struct {
	context.Context
	mu        sync.Mutex
	remaining int
	done      chan struct{}
	canceled  bool
}

func newCancelAfterContext(checks int) *cancelAfterContext {
	return &cancelAfterContext{Context: context.Background(), remaining: checks, done: make(chan struct{})}
}

func (c *cancelAfterContext) Done() <-chan struct{} { return c.done }

func (c *cancelAfterContext) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.canceled {
		return context.Canceled
	}
	c.remaining--
	if c.remaining <= 0 {
		close(c.done)
		c.canceled = true
		return context.Canceled
	}
	return nil
}

func mustUpsert(t *testing.T, idx indexcontract.Index, items []*retrieval.Result) {
	t.Helper()
	if err := idx.Upsert(context.Background(), items); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
}

func mustSearch(t *testing.T, idx indexcontract.Index, query indexcontract.IndexQuery) []*retrieval.Result {
	t.Helper()
	got, err := idx.Search(context.Background(), query)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	return got
}

func resultIDs(results []*retrieval.Result) string {
	if len(results) == 0 {
		return ""
	}
	ids := results[0].ID
	for _, result := range results[1:] {
		ids += "," + result.ID
	}
	return ids
}

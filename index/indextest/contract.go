// Package indextest provides a reusable contract suite for Index implementations.
package indextest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
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

	t.Run("delete and empty replacement preserve index space", func(t *testing.T) {
		actions := []struct {
			name string
			run  func(indexcontract.Index) error
		}{
			{name: "delete all", run: func(idx indexcontract.Index) error {
				return idx.DeleteDocument(context.Background(), "doc-a")
			}},
			{name: "empty replacement", run: func(idx indexcontract.Index) error {
				return idx.ReplaceDocuments(context.Background(), []indexcontract.DocumentReplacement{{DocumentID: "doc-a"}})
			}},
		}
		for _, action := range actions {
			t.Run(action.name, func(t *testing.T) {
				idx := newIndex()
				mustUpsert(t, idx, []*retrieval.Result{{ID: "a", DocumentID: "doc-a", IndexIdentity: "old", Embedding: []float64{1, 0}}})
				if err := action.run(idx); err != nil {
					t.Fatalf("empty index action: %v", err)
				}
				if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Identity: "new"}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
					t.Fatalf("empty latched identity error = %v", err)
				}
				if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Identity: "old", Vector: []float32{1, 0, 0}}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
					t.Fatalf("empty latched dimension error = %v", err)
				}
				if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "b", IndexIdentity: "new", Embedding: []float64{1, 0}}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
					t.Fatalf("empty latched write identity error = %v", err)
				}
				if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "b", IndexIdentity: "old", Embedding: []float64{1, 0, 0}}}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
					t.Fatalf("empty latched write dimension error = %v", err)
				}
			})
		}
	})

	t.Run("delete rejects blank document IDs without mutation", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{
			{ID: "blank", DocumentID: "", Content: "preserved blank", Embedding: []float64{1, 0}},
			{ID: "space", DocumentID: " \t\n", Content: "preserved whitespace", Embedding: []float64{1, 0}},
			{ID: "valid", DocumentID: "doc-a", Content: "preserved valid", Embedding: []float64{1, 0}},
		})
		for _, documentID := range []string{"", " \t\n"} {
			if err := idx.DeleteDocument(context.Background(), documentID); !errors.Is(err, indexcontract.ErrInvalidDocumentID) {
				t.Fatalf("DeleteDocument(%q) error = %v, want %v", documentID, err, indexcontract.ErrInvalidDocumentID)
			}
			got := mustSearch(t, idx, indexcontract.IndexQuery{Vector: []float32{1, 0}})
			if ids := resultIDs(got); ids != "blank,space,valid" {
				t.Fatalf("state after DeleteDocument(%q) = %#v", documentID, got)
			}
		}
	})

	t.Run("delete uses exact document ownership", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{
			{ID: "doc-a#0", DocumentID: "doc-a", Embedding: []float64{1, 0}},
			{ID: "doc-a#legacy", Content: "unowned", Embedding: []float64{1, 0}},
			{ID: "doc-a#1#0", Content: "nested unowned", Embedding: []float64{1, 0}},
			{ID: "other", DocumentID: "doc-b", Embedding: []float64{1, 0}},
		})
		if err := idx.DeleteDocument(context.Background(), "doc-a"); err != nil {
			t.Fatalf("DeleteDocument: %v", err)
		}
		got := mustSearch(t, idx, indexcontract.IndexQuery{Vector: []float32{1, 0}})
		if ids := resultIDs(got); ids != "doc-a#1#0,doc-a#legacy,other" {
			t.Fatalf("DeleteDocument exact ownership IDs = %s, want doc-a#1#0,doc-a#legacy,other", ids)
		}
	})

	t.Run("replace complete document revisions", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{
			{ID: "doc-a#0", DocumentID: "doc-a", Content: "old zero", Embedding: []float64{1, 0}},
			{ID: "doc-a#1", DocumentID: "doc-a", Content: "old one", Embedding: []float64{0, 1}},
			{ID: "doc-a#legacy", Content: "unowned", Embedding: []float64{1, 0}},
			{ID: "doc-b#0", DocumentID: "doc-b", Content: "retained", Embedding: []float64{1, 0}},
		})
		if err := idx.ReplaceDocuments(context.Background(), []indexcontract.DocumentReplacement{{
			DocumentID: "doc-a",
			Results:    []*retrieval.Result{{ID: "doc-a#0", DocumentID: "doc-a", Content: "new", Embedding: []float64{1, 0}}},
		}}); err != nil {
			t.Fatalf("ReplaceDocuments shorter revision: %v", err)
		}
		got := mustSearch(t, idx, indexcontract.IndexQuery{})
		if ids := resultIDs(got); ids != "doc-a#0,doc-a#legacy,doc-b#0" || got[0].Content != "new" {
			t.Fatalf("shorter replacement = %#v", got)
		}
		if err := idx.ReplaceDocuments(context.Background(), []indexcontract.DocumentReplacement{{DocumentID: "doc-a"}}); err != nil {
			t.Fatalf("ReplaceDocuments empty revision: %v", err)
		}
		got = mustSearch(t, idx, indexcontract.IndexQuery{})
		if ids := resultIDs(got); ids != "doc-a#legacy,doc-b#0" {
			t.Fatalf("empty replacement IDs = %s, want doc-a#legacy,doc-b#0", ids)
		}
	})

	t.Run("replace batch validates and rolls back atomically", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{
			{ID: "doc-a#0", DocumentID: "doc-a", Content: "old a", IndexIdentity: "space", Embedding: []float64{1, 0}},
			{ID: "doc-b#0", DocumentID: "doc-b", Content: "old b", IndexIdentity: "space", Embedding: []float64{0, 1}},
		})
		invalidBatches := []struct {
			name         string
			replacements []indexcontract.DocumentReplacement
			want         error
		}{
			{name: "blank", replacements: []indexcontract.DocumentReplacement{{DocumentID: " "}}, want: indexcontract.ErrInvalidDocumentID},
			{name: "duplicate", replacements: []indexcontract.DocumentReplacement{{DocumentID: "doc-a"}, {DocumentID: "doc-a"}}, want: indexcontract.ErrDuplicateDocumentID},
			{name: "duplicate result ID", replacements: []indexcontract.DocumentReplacement{
				{DocumentID: "doc-a", Results: []*retrieval.Result{{ID: "same", DocumentID: "doc-a", IndexIdentity: "space", Embedding: []float64{1, 0}}}},
				{DocumentID: "doc-b", Results: []*retrieval.Result{{ID: "same", DocumentID: "doc-b", IndexIdentity: "space", Embedding: []float64{1, 0}}}},
			}, want: indexcontract.ErrResultIDConflict},
			{name: "retained result ID", replacements: []indexcontract.DocumentReplacement{
				{DocumentID: "doc-a", Results: []*retrieval.Result{{ID: "doc-b#0", DocumentID: "doc-a", IndexIdentity: "space", Embedding: []float64{1, 0}}}},
			}, want: indexcontract.ErrResultIDConflict},
			{name: "dimension", replacements: []indexcontract.DocumentReplacement{
				{DocumentID: "doc-a", Results: []*retrieval.Result{{ID: "doc-a#0", DocumentID: "doc-a", Content: "new a", IndexIdentity: "space", Embedding: []float64{1, 0, 0}}}},
				{DocumentID: "doc-b", Results: []*retrieval.Result{{ID: "doc-b#0", DocumentID: "doc-b", Content: "new b", IndexIdentity: "space", Embedding: []float64{1, 0}}}},
			}, want: indexcontract.ErrDimensionMismatch},
			{name: "identity", replacements: []indexcontract.DocumentReplacement{
				{DocumentID: "doc-a", Results: []*retrieval.Result{{ID: "doc-a#0", DocumentID: "doc-a", Content: "new a", IndexIdentity: "other", Embedding: []float64{1, 0}}}},
			}, want: indexcontract.ErrIdentityMismatch},
		}
		for _, tc := range invalidBatches {
			t.Run(tc.name, func(t *testing.T) {
				if err := idx.ReplaceDocuments(context.Background(), tc.replacements); !errors.Is(err, tc.want) {
					t.Fatalf("ReplaceDocuments error = %v, want %v", err, tc.want)
				}
				got := mustSearch(t, idx, indexcontract.IndexQuery{Identity: "space"})
				if ids := resultIDs(got); ids != "doc-a#0,doc-b#0" || got[0].Content != "old a" || got[1].Content != "old b" {
					t.Fatalf("state changed after failed replacement: %#v", got)
				}
			})
		}
	})

	t.Run("replace all documents rejects a new vector space and rolls back", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{{ID: "old#0", DocumentID: "old", Content: "preserved", IndexIdentity: "old", Embedding: []float64{1, 0}}})
		err := idx.ReplaceDocuments(context.Background(), []indexcontract.DocumentReplacement{{
			DocumentID: "old",
			Results:    []*retrieval.Result{{ID: "old#0", DocumentID: "old", IndexIdentity: "new", Embedding: []float64{1, 0, 0}}},
		}})
		if !errors.Is(err, indexcontract.ErrIdentityMismatch) {
			t.Fatalf("ReplaceDocuments identity error = %v", err)
		}
		err = idx.ReplaceDocuments(context.Background(), []indexcontract.DocumentReplacement{{
			DocumentID: "old",
			Results:    []*retrieval.Result{{ID: "old#0", DocumentID: "old", IndexIdentity: "old", Embedding: []float64{1, 0, 0}}},
		}})
		if !errors.Is(err, indexcontract.ErrDimensionMismatch) {
			t.Fatalf("ReplaceDocuments dimension error = %v", err)
		}
		got := mustSearch(t, idx, indexcontract.IndexQuery{Identity: "old", Vector: []float32{1, 0}})
		if len(got) != 1 || got[0].Content != "preserved" {
			t.Fatalf("state after rejected replacement = %#v", got)
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
			{ID: "a", DocumentID: "doc-a", Filename: "a.md", Embedding: []float64{1, 0}, Metadata: map[string]any{"tenant": "one", "active": true, "literal.key": "value", `literal"key`: "quoted", "array[0]": "bracketed", "nullable": nil}},
			{ID: "b", DocumentID: "doc-b", Filename: "b.md", Embedding: []float64{1, 0}, Metadata: map[string]any{"tenant": "two", "active": false, "literal": map[string]any{"key": "value"}, `literal"key`: "other", "array": []any{"bracketed"}, "nullable": "set"}},
			{ID: "c", DocumentID: "doc-c", Filename: "c.md", Embedding: []float64{1, 0}, Metadata: map[string]any{"tenant": "three"}},
		})
		filters := []map[string]any{
			{"id": "a"}, {"document_id": "doc-a"}, {"filename": "a.md"},
			{"tenant": "one"}, {"tenant": "one", "active": true},
			{"literal.key": "value"}, {`literal"key`: "quoted"}, {"array[0]": "bracketed"}, {"nullable": nil},
		}
		for _, filter := range filters {
			got := mustSearch(t, idx, indexcontract.IndexQuery{Filter: filter})
			if len(got) != 1 || got[0].ID != "a" {
				t.Fatalf("filter %#v = %#v", filter, got)
			}
		}
	})

	t.Run("JSON scalar filter semantics", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{
			{ID: "exact", Metadata: map[string]any{"one": 1, "large": uint64(9007199254740993), "fraction": 1.5, "nullable": nil, "string": "1", "boolean": true}},
			{ID: "near", Metadata: map[string]any{"one": 2, "large": uint64(9007199254740992), "fraction": -1.5, "string": "true", "boolean": false}},
		})
		oneValues := []any{
			int(1), int8(1), int16(1), int32(1), int64(1),
			uint(1), uint8(1), uint16(1), uint32(1), uint64(1),
			float32(1), float64(1), json.Number("1.0"),
		}
		for _, value := range oneValues {
			got := mustSearch(t, idx, indexcontract.IndexQuery{Filter: map[string]any{"one": value}})
			if len(got) != 1 || got[0].ID != "exact" {
				t.Fatalf("numeric filter %T(%v) = %#v", value, value, got)
			}
		}
		matches := []map[string]any{
			{"large": uint64(9007199254740993)},
			{"large": json.Number("9007199254740993")},
			{"fraction": 1.5},
			{"nullable": nil},
			{"string": "1"},
			{"boolean": true},
		}
		for _, filter := range matches {
			got := mustSearch(t, idx, indexcontract.IndexQuery{Filter: filter})
			if len(got) != 1 || got[0].ID != "exact" {
				t.Fatalf("filter %#v = %#v", filter, got)
			}
		}
		noMatches := []map[string]any{
			{"id": "exact", "large": uint64(9007199254740992)},
			{"id": "exact", "fraction": -1.5},
			{"one": -1},
			{"missing": nil},
			{"string": 1},
			{"boolean": "true"},
			{"id": 1},
		}
		for _, filter := range noMatches {
			got := mustSearch(t, idx, indexcontract.IndexQuery{Filter: filter})
			if len(got) != 0 {
				t.Fatalf("filter %#v unexpectedly matched %#v", filter, got)
			}
		}
		for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Filter: map[string]any{"one": value}}); err == nil {
				t.Fatalf("non-finite filter %v did not fail", value)
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
		if err := idx.ReplaceDocuments(canceled, []indexcontract.DocumentReplacement{{DocumentID: "doc"}}); !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-canceled ReplaceDocuments error = %v", err)
		}
		if err := idx.DeleteDocument(canceled, ""); !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-canceled DeleteDocument error = %v", err)
		}
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

	t.Run("non-finite vectors are rejected without mutation or latching", func(t *testing.T) {
		values := []struct {
			name  string
			value float64
		}{
			{name: "NaN", value: math.NaN()},
			{name: "positive infinity", value: math.Inf(1)},
			{name: "negative infinity", value: math.Inf(-1)},
		}
		for _, tc := range values {
			t.Run(tc.name, func(t *testing.T) {
				idx := newIndex()
				mustUpsert(t, idx, []*retrieval.Result{{ID: "kept", DocumentID: "doc", Content: "old", IndexIdentity: "space", Embedding: []float64{1, 0}}})

				if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "bad-upsert", IndexIdentity: "space", Embedding: []float64{tc.value, 0}}}); err == nil || !strings.Contains(err.Error(), "must be finite") {
					t.Fatalf("Upsert non-finite error = %v", err)
				}
				if err := idx.ReplaceDocuments(context.Background(), []indexcontract.DocumentReplacement{{
					DocumentID: "doc",
					Results:    []*retrieval.Result{{ID: "replacement", DocumentID: "doc", Content: "new", IndexIdentity: "space", Embedding: []float64{tc.value, 0}}},
				}}); err == nil || !strings.Contains(err.Error(), "must be finite") {
					t.Fatalf("ReplaceDocuments non-finite error = %v", err)
				}
				if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Identity: "space", Vector: []float32{float32(tc.value), 0}}); err == nil || !strings.Contains(err.Error(), "must be finite") {
					t.Fatalf("Search non-finite error = %v", err)
				}
				got := mustSearch(t, idx, indexcontract.IndexQuery{Identity: "space", Vector: []float32{1, 0}})
				if len(got) != 1 || got[0].ID != "kept" || got[0].Content != "old" {
					t.Fatalf("state after rejected vectors = %#v", got)
				}

				freshUpsert := newIndex()
				if err := freshUpsert.Upsert(context.Background(), []*retrieval.Result{{ID: "bad", IndexIdentity: "bad", Embedding: []float64{tc.value, 0}}}); err == nil {
					t.Fatal("fresh non-finite Upsert succeeded")
				}
				mustUpsert(t, freshUpsert, []*retrieval.Result{{ID: "good", IndexIdentity: "good", Embedding: []float64{1, 0, 0}}})

				freshReplacement := newIndex()
				if err := freshReplacement.ReplaceDocuments(context.Background(), []indexcontract.DocumentReplacement{{
					DocumentID: "bad",
					Results:    []*retrieval.Result{{ID: "bad", DocumentID: "bad", IndexIdentity: "bad", Embedding: []float64{tc.value, 0}}},
				}}); err == nil {
					t.Fatal("fresh non-finite ReplaceDocuments succeeded")
				}
				if err := freshReplacement.ReplaceDocuments(context.Background(), []indexcontract.DocumentReplacement{{
					DocumentID: "good",
					Results:    []*retrieval.Result{{ID: "good", DocumentID: "good", IndexIdentity: "good", Embedding: []float64{1, 0, 0}}},
				}}); err != nil {
					t.Fatalf("valid replacement after rejected latch: %v", err)
				}
			})
		}
	})

	t.Run("vector validation preserves identity and dimension precedence", func(t *testing.T) {
		idx := newIndex()
		mustUpsert(t, idx, []*retrieval.Result{{ID: "kept", IndexIdentity: "space", Embedding: []float64{1, 0}}})
		if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "bad", IndexIdentity: "other", Embedding: []float64{math.NaN(), 0, 0}}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
			t.Fatalf("Upsert precedence error = %v, want identity mismatch", err)
		}
		if err := idx.Upsert(context.Background(), []*retrieval.Result{{ID: "bad", IndexIdentity: "space", Embedding: []float64{math.NaN(), 0, 0}}}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
			t.Fatalf("Upsert precedence error = %v, want dimension mismatch", err)
		}
		if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Identity: "other", Vector: []float32{float32(math.NaN()), 0, 0}}); !errors.Is(err, indexcontract.ErrIdentityMismatch) {
			t.Fatalf("Search precedence error = %v, want identity mismatch", err)
		}
		if _, err := idx.Search(context.Background(), indexcontract.IndexQuery{Identity: "space", Vector: []float32{float32(math.NaN()), 0, 0}}); !errors.Is(err, indexcontract.ErrDimensionMismatch) {
			t.Fatalf("Search precedence error = %v, want dimension mismatch", err)
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
		mustUpsert(t, idx, []*retrieval.Result{{ID: "new", IndexIdentity: "new", Embedding: []float64{1, 0, 0}}})
		got := mustSearch(t, idx, indexcontract.IndexQuery{Identity: "new", Vector: []float32{1, 0, 0}})
		if len(got) != 1 || got[0].ID != "new" {
			t.Fatalf("Search after Reset = %#v", got)
		}
	})

	t.Run("mutation isolation and round trip", func(t *testing.T) {
		idx := newIndex()
		item := &retrieval.Result{ID: "stable", DocumentID: "doc", Filename: "doc.md", Content: "original", Embedding: []float64{1, 0}, Metadata: map[string]any{
			"tenant": "one",
			"nested": map[string]any{"value": "original"},
			"array":  []any{map[string]any{"value": "original"}},
		}}
		mustUpsert(t, idx, []*retrieval.Result{item})
		item.Content = "input mutation"
		item.Embedding[0] = 0
		item.Metadata["tenant"] = "changed"
		item.Metadata["nested"].(map[string]any)["value"] = "input mutation"
		item.Metadata["array"].([]any)[0].(map[string]any)["value"] = "input mutation"

		got := mustSearch(t, idx, indexcontract.IndexQuery{Filter: map[string]any{"id": "stable"}})
		if len(got) != 1 || got[0].ID != "stable" || got[0].DocumentID != "doc" || got[0].Filename != "doc.md" || got[0].Content != "original" || got[0].Embedding[0] != 1 || got[0].Metadata["tenant"] != "one" || got[0].Metadata["nested"].(map[string]any)["value"] != "original" || got[0].Metadata["array"].([]any)[0].(map[string]any)["value"] != "original" {
			t.Fatalf("round trip after input mutation = %#v", got)
		}
		got[0].Embedding[0] = 0
		got[0].Metadata["tenant"] = "returned mutation"
		got[0].Metadata["nested"].(map[string]any)["value"] = "returned mutation"
		got[0].Metadata["array"].([]any)[0].(map[string]any)["value"] = "returned mutation"
		again := mustSearch(t, idx, indexcontract.IndexQuery{Filter: map[string]any{"id": "stable"}})
		if again[0].Embedding[0] != 1 || again[0].Metadata["tenant"] != "one" || again[0].Metadata["nested"].(map[string]any)["value"] != "original" || again[0].Metadata["array"].([]any)[0].(map[string]any)["value"] != "original" {
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

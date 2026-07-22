package indexutil

import (
	"context"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestSearch(t *testing.T) {
	t.Parallel()

	item1 := &retrieval.Result{
		ID:            "doc1#1",
		DocumentID:    "doc1",
		Filename:      "file1.txt",
		Content:       "go programming language",
		Embedding:     []float64{1.0, 0.0},
		Metadata:      map[string]any{"env": "prod"},
		CombinedScore: 0.9,
	}
	item2 := &retrieval.Result{
		ID:            "doc1#2",
		DocumentID:    "doc1",
		Filename:      "file1.txt",
		Content:       "python programming language",
		Embedding:     []float64{0.0, 1.0},
		Metadata:      map[string]any{"env": "dev"},
		CombinedScore: 0.9,
	}
	item3 := &retrieval.Result{
		ID:            "doc2#1",
		DocumentID:    "doc2",
		Filename:      "file2.txt",
		Content:       "rust programming language",
		Embedding:     []float64{0.5, 0.5},
		Metadata:      map[string]any{"env": "prod"},
		CombinedScore: 0.8,
	}

	items := []*retrieval.Result{nil, item1, item2, item3}

	t.Run("empty items returns nil", func(t *testing.T) {
		res, err := Search(context.Background(), indexcontract.IndexQuery{}, nil)
		if err != nil || res != nil {
			t.Fatalf("expected nil, got %v, %v", res, err)
		}
	})

	t.Run("context cancelled returns error", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := Search(ctx, indexcontract.IndexQuery{}, items)
		if err == nil {
			t.Fatal("expected context error, got nil")
		}
	})

	t.Run("filter matching metadata, document_id, and id", func(t *testing.T) {
		query := indexcontract.IndexQuery{
			Filter: map[string]any{"env": "prod"},
		}
		res, err := Search(context.Background(), query, items)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 2 {
			t.Fatalf("expected 2 items, got %d", len(res))
		}
	})

	t.Run("filter by ID and document_id and filename", func(t *testing.T) {
		query := indexcontract.IndexQuery{
			Filter: map[string]any{
				"id":          "doc1#1",
				"document_id": "doc1",
				"filename":    "file1.txt",
			},
		}
		res, err := Search(context.Background(), query, items)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 1 || res[0].ID != "doc1#1" {
			t.Fatalf("expected doc1#1, got %v", res)
		}
	})

	t.Run("dimension mismatch error", func(t *testing.T) {
		query := indexcontract.IndexQuery{
			Vector: []float32{1.0, 0.0, 0.0},
		}
		_, err := Search(context.Background(), query, items)
		if err == nil {
			t.Fatal("expected dimension mismatch error, got nil")
		}
	})

	t.Run("limit truncates results and score tie-breaker orders by ID", func(t *testing.T) {
		query := indexcontract.IndexQuery{
			Limit: 1,
		}
		res, err := Search(context.Background(), query, items)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		if len(res) != 1 {
			t.Fatalf("expected 1 item due to limit, got %d", len(res))
		}
	})
}

func TestClone(t *testing.T) {
	t.Parallel()

	orig := &retrieval.Result{
		ID:        "1",
		Embedding: []float64{1.0, 2.0},
		Metadata: map[string]any{
			"key": "val",
			"object": map[string]any{
				"nested": map[string]any{"value": "original"},
			},
			"array": []any{
				"first",
				map[string]any{"value": "original"},
				[]any{"nested", map[string]any{"value": "original"}},
			},
			"nil":        nil,
			"nil_object": map[string]any(nil),
			"nil_array":  []any(nil),
		},
		Explain: &retrieval.SearchExplanation{
			HybridRank: 1,
			FinalRank:  2,
		},
	}

	cloned := Clone(orig)
	if cloned == orig {
		t.Fatal("expected different pointer")
	}
	if cloned.Explain != nil {
		t.Fatalf("clone explanation = %#v, want nil", cloned.Explain)
	}
	if orig.Explain == nil || orig.Explain.FinalRank != 2 {
		t.Fatalf("source explanation changed: %#v", orig.Explain)
	}

	cloned.Embedding[0] = 99.0
	if orig.Embedding[0] == 99.0 {
		t.Fatal("embedding slice aliased")
	}

	cloned.Metadata["key"] = "new_val"
	if orig.Metadata["key"] == "new_val" {
		t.Fatal("metadata map aliased")
	}
	if cloned.Metadata["key"] != "new_val" {
		t.Fatalf("clone metadata = %#v, want preserved writable copy", cloned.Metadata)
	}

	cloned.Metadata["object"].(map[string]any)["nested"].(map[string]any)["value"] = "changed"
	cloned.Metadata["array"].([]any)[1].(map[string]any)["value"] = "changed"
	cloned.Metadata["array"].([]any)[2].([]any)[1].(map[string]any)["value"] = "changed"
	if got := orig.Metadata["object"].(map[string]any)["nested"].(map[string]any)["value"]; got != "original" {
		t.Fatalf("nested object aliased: got %v", got)
	}
	if got := orig.Metadata["array"].([]any)[1].(map[string]any)["value"]; got != "original" {
		t.Fatalf("object in array aliased: got %v", got)
	}
	if got := orig.Metadata["array"].([]any)[2].([]any)[1].(map[string]any)["value"]; got != "original" {
		t.Fatalf("nested array aliased: got %v", got)
	}
	if cloned.Metadata["nil"] != nil {
		t.Fatalf("nil metadata value = %#v, want nil", cloned.Metadata["nil"])
	}
	if cloned.Metadata["nil_object"].(map[string]any) != nil {
		t.Fatalf("nil object = %#v, want typed nil", cloned.Metadata["nil_object"])
	}
	if cloned.Metadata["nil_array"].([]any) != nil {
		t.Fatalf("nil array = %#v, want typed nil", cloned.Metadata["nil_array"])
	}

	origNoMeta := &retrieval.Result{ID: "2"}
	clonedNoMeta := Clone(origNoMeta)
	if clonedNoMeta.Metadata != nil {
		t.Fatal("expected nil metadata in clone")
	}
}

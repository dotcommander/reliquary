package inmem

import (
	"context"
	"errors"
	"testing"

	indexcontract "github.com/dotcommander/reliquary/index"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestSearchSnapshotsStoredResultsWithoutExposingThem(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	idx := New()
	input := &retrieval.Result{
		ID: "chunk", DocumentID: "doc", IndexIdentity: "space", Content: "original",
		Embedding: []float64{1, 0},
		Metadata:  map[string]any{"nested": map[string]any{"values": []any{"original"}}},
	}
	if err := idx.Upsert(ctx, []*retrieval.Result{input}); err != nil {
		t.Fatal(err)
	}
	input.Embedding[0] = 9
	input.Metadata["nested"].(map[string]any)["values"].([]any)[0] = "input changed"

	query := indexcontract.IndexQuery{Identity: "space", Vector: []float32{1, 0}}
	first, err := idx.Search(ctx, query)
	if err != nil || len(first) != 1 {
		t.Fatalf("first search: results=%v err=%v", first, err)
	}
	if first[0] == idx.items["chunk"] {
		t.Fatal("search exposed stored result pointer")
	}
	first[0].Content = "changed"
	first[0].Embedding[0] = 7
	first[0].Metadata["nested"].(map[string]any)["values"].([]any)[0] = "changed"

	second, err := idx.Search(ctx, query)
	if err != nil || len(second) != 1 {
		t.Fatalf("second search: results=%v err=%v", second, err)
	}
	got := second[0]
	if got.Content != "original" || got.Embedding[0] != 1 || got.Metadata["nested"].(map[string]any)["values"].([]any)[0] != "original" {
		t.Fatalf("search result aliased input or previous result: %+v", got)
	}
	if idx.items["chunk"].EmbeddingScore != 0 || idx.items["chunk"].CombinedScore != 0 {
		t.Fatal("search scoring mutated stored result")
	}
}

func TestReplaceDocumentsRetainsOwnedResultsAndClonesIncoming(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	idx := New()
	retained := &retrieval.Result{ID: "retained", DocumentID: "keep", IndexIdentity: "space", Embedding: []float64{1}, Metadata: map[string]any{"value": []any{"keep"}}}
	old := &retrieval.Result{ID: "old", DocumentID: "replace", IndexIdentity: "space", Embedding: []float64{1}}
	if err := idx.Upsert(ctx, []*retrieval.Result{retained, old}); err != nil {
		t.Fatal(err)
	}
	owned := idx.items["retained"]
	incoming := &retrieval.Result{ID: "new", DocumentID: "replace", IndexIdentity: "space", Embedding: []float64{1}, Metadata: map[string]any{"value": []any{"new"}}}
	if err := idx.ReplaceDocuments(ctx, []indexcontract.DocumentReplacement{{DocumentID: "replace", Results: []*retrieval.Result{incoming}}}); err != nil {
		t.Fatal(err)
	}
	if idx.items["retained"] != owned || idx.items["old"] != nil {
		t.Fatal("replacement did not reuse retained owned result or remove old revision")
	}
	if idx.items["new"] == incoming {
		t.Fatal("replacement stored caller-owned result")
	}
	retained.Embedding[0] = 8
	retained.Metadata["value"].([]any)[0] = "changed"
	incoming.Embedding[0] = 9
	incoming.Metadata["value"].([]any)[0] = "changed"
	if owned.Embedding[0] != 1 || owned.Metadata["value"].([]any)[0] != "keep" || idx.items["new"].Embedding[0] != 1 || idx.items["new"].Metadata["value"].([]any)[0] != "new" {
		t.Fatal("stored results aliased caller-owned inputs")
	}
	conflict := &retrieval.Result{ID: "retained", DocumentID: "replace", IndexIdentity: "space", Embedding: []float64{1}}
	if err := idx.ReplaceDocuments(ctx, []indexcontract.DocumentReplacement{{DocumentID: "replace", Results: []*retrieval.Result{conflict}}}); !errors.Is(err, indexcontract.ErrResultIDConflict) {
		t.Fatalf("retained ID collision: %v", err)
	}
	if idx.items["retained"] != owned || idx.items["new"] == nil {
		t.Fatal("failed replacement changed stored results")
	}
}

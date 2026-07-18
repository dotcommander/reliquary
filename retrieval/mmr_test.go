package retrieval

import "testing"

func TestMMRImprovesDiversity(t *testing.T) {
	t.Parallel()

	items := []MMRItem{
		{ID: "a1", Score: 0.99, Embedding: []float64{1, 0}, Topic: "a"},
		{ID: "a2", Score: 0.98, Embedding: []float64{0.99, 0.01}, Topic: "a"},
		{ID: "b1", Score: 0.90, Embedding: []float64{0, 1}, Topic: "b"},
	}

	got := MMR(items, 2, 0.5)
	if len(got) != 2 {
		t.Fatalf("MMR() returned %d items, want 2", len(got))
	}
	if got[0].ID == got[1].ID {
		t.Fatalf("MMR returned duplicate IDs")
	}
	if got[0].Topic == got[1].Topic {
		t.Fatalf("MMR did not diversify topics")
	}
}

// TestMMRDuplicateIDs pins that MMR does NOT deduplicate by ID — two items
// sharing an ID are both selectable. With lambda=1 (pure relevance) both
// "dup" items outrank the lower-scored unique item.
func TestMMRDuplicateIDs(t *testing.T) {
	t.Parallel()

	items := []MMRItem{
		{ID: "dup", Score: 0.9, Embedding: []float64{1, 0}},
		{ID: "dup", Score: 0.8, Embedding: []float64{1, 0}},
		{ID: "other", Score: 0.1, Embedding: []float64{0, 1}},
	}
	got := MMR(items, 2, 1)
	if len(got) != 2 {
		t.Fatalf("MMR() returned %d items, want 2", len(got))
	}
	if got[0].ID != "dup" || got[1].ID != "dup" {
		t.Fatalf("MMR selected IDs = (%q, %q), want both \"dup\" (no dedup)", got[0].ID, got[1].ID)
	}
	if got[0].Score != 0.9 || got[1].Score != 0.8 {
		t.Fatalf("MMR selected scores = (%v, %v), want (0.9, 0.8)", got[0].Score, got[1].Score)
	}
}

// TestMMREmptyEmbeddingTreatedAsZeroSimilarity pins that an item with an empty
// embedding contributes 0 to maxSimilarity, so with lambda=0.5 the second pick
// is driven purely by Score among the zero-similarity candidates. All three
// items have empty embeddings, so similarity is always 0 and MMR reduces to
// descending Score order.
func TestMMREmptyEmbeddingTreatedAsZeroSimilarity(t *testing.T) {
	t.Parallel()

	items := []MMRItem{
		{ID: "a", Score: 0.9},
		{ID: "b", Score: 0.7},
		{ID: "c", Score: 0.5},
	}
	got := MMR(items, 3, 0.5)
	if len(got) != 3 {
		t.Fatalf("MMR() returned %d items, want 3", len(got))
	}
	if got[0].ID != "a" || got[1].ID != "b" || got[2].ID != "c" {
		t.Fatalf("MMR order = (%q, %q, %q), want (a, b, c) by descending score", got[0].ID, got[1].ID, got[2].ID)
	}
}

// TestMMRTieBreakEarliestIndexWins pins the strict-greater (`score > bestScore`)
// comparison: when two candidates yield equal MMR score, the earlier index is
// selected first. Two items with identical Score and identical embedding tie on
// every metric; the input-order-earlier one must come first.
func TestMMRTieBreakEarliestIndexWins(t *testing.T) {
	t.Parallel()

	items := []MMRItem{
		{ID: "first", Score: 0.5, Embedding: []float64{1, 0}},
		{ID: "second", Score: 0.5, Embedding: []float64{1, 0}},
	}
	got := MMR(items, 2, 0.5)
	if len(got) != 2 {
		t.Fatalf("MMR() returned %d items, want 2", len(got))
	}
	if got[0].ID != "first" || got[1].ID != "second" {
		t.Fatalf("MMR tie order = (%q, %q), want (first, second) — earliest index wins", got[0].ID, got[1].ID)
	}
}

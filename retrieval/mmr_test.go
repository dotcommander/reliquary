package retrieval

import "testing"

func TestMMRAndDiversify(t *testing.T) {
	t.Parallel()

	r1 := &Result{ID: "r1", CombinedScore: 0.9, Embedding: []float64{1, 0}, Filename: "f1"}
	r2 := &Result{ID: "r2", CombinedScore: 0.85, Embedding: []float64{0.99, 0.01}, Filename: "f1"}
	r3 := &Result{ID: "r3", CombinedScore: 0.7, Embedding: []float64{0, 1}, Filename: "f2"}

	results := []*Result{nil, r1, r2, r3}

	t.Run("MMRItems filters nil", func(t *testing.T) {
		items := MMRItems(results)
		if len(items) != 3 {
			t.Fatalf("len(items) = %d, want 3", len(items))
		}
		if items[0].ID != "r1" || items[0].Topic != "f1" {
			t.Fatalf("item 0 = %#v", items[0])
		}
	})

	t.Run("Diversify selects diverse results", func(t *testing.T) {
		div := Diversify(results, 2, 0.5)
		if len(div) != 2 {
			t.Fatalf("len(div) = %d, want 2", len(div))
		}
		// r1 should be chosen first, and r3 should be preferred over r2 due to diversity (different direction)
		if div[0].ID != "r1" {
			t.Fatalf("first = %s, want r1", div[0].ID)
		}
		if div[1].ID != "r3" {
			t.Fatalf("second = %s, want r3", div[1].ID)
		}
	})

	t.Run("MMR empty edge cases", func(t *testing.T) {
		if got := MMR(nil, 5, 0.5); got != nil {
			t.Fatalf("MMR(nil) = %v, want nil", got)
		}
		items := MMRItems([]*Result{r1})
		if got := MMR(items, 0, 0.5); got != nil {
			t.Fatalf("MMR(k=0) = %v, want nil", got)
		}
	})

	t.Run("maxSimilarity handles missing embeddings", func(t *testing.T) {
		itemNoEmbed := MMRItem{ID: "no_embed", Score: 0.5}
		itemWithEmbed := MMRItem{ID: "embed", Score: 0.8, Embedding: []float64{1, 0}}

		sim1 := maxSimilarity(itemNoEmbed, []MMRItem{itemWithEmbed})
		if sim1 != 0 {
			t.Fatalf("sim1 = %v, want 0", sim1)
		}

		sim2 := maxSimilarity(itemWithEmbed, []MMRItem{itemNoEmbed})
		if sim2 != 0 {
			t.Fatalf("sim2 = %v, want 0", sim2)
		}
	})
}

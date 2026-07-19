package retrieval

import "testing"

func TestNewScorerOpts(t *testing.T) {
	t.Parallel()

	scorer := NewScorerOpts(
		Embedding(0.5),
		Keyword(0.2),
		Filename(0.1),
		Recency(0.1),
		Importance(0.1),
	)

	w := scorer.weights
	if w.Embedding != 0.5 {
		t.Fatalf("Embedding weight = %v, want 0.5", w.Embedding)
	}
	if w.Keyword != 0.2 {
		t.Fatalf("Keyword weight = %v, want 0.2", w.Keyword)
	}
	if w.Filename != 0.1 {
		t.Fatalf("Filename weight = %v, want 0.1", w.Filename)
	}
	if w.Recency != 0.1 {
		t.Fatalf("Recency weight = %v, want 0.1", w.Recency)
	}
	if w.Importance != 0.1 {
		t.Fatalf("Importance weight = %v, want 0.1", w.Importance)
	}
}

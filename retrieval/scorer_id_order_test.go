package retrieval

import (
	"testing"

	"github.com/dotcommander/reliquary/embedding"
)

func TestRerankEmbeddingByIDPreservesScoringAndLegacyTieOrder(t *testing.T) {
	t.Parallel()

	makeResults := func() []*Result {
		return []*Result{
			{ID: "b", Content: "alpha", Embedding: []float64{1, 0}},
			{ID: "c", Content: "bravo", Embedding: []float64{0, 1}},
			{ID: "a", Content: "alpha", Embedding: []float64{1, 0}},
		}
	}
	scorer := NewScorer(DefaultWeights())
	query := embedding.Vector{1, 0}
	legacy := scorer.RerankEmbedding(query, "alpha", makeResults())
	plain := scorer.Rerank(EmbeddingVector(query), "alpha", makeResults())
	byID := scorer.RerankEmbeddingByID(query, "alpha", makeResults())

	for _, ranked := range [][]*Result{legacy, plain} {
		if ranked[0].ID != "b" || ranked[1].ID != "a" || ranked[2].ID != "c" {
			t.Fatalf("score-only order = %q, %q, %q, want b, a, c", ranked[0].ID, ranked[1].ID, ranked[2].ID)
		}
	}
	if byID[0].ID != "a" || byID[1].ID != "b" || byID[2].ID != "c" {
		t.Fatalf("score/ID order = %q, %q, %q, want a, b, c", byID[0].ID, byID[1].ID, byID[2].ID)
	}
	for _, ranked := range [][]*Result{legacy, plain, byID} {
		if ranked[0].CombinedScore != ranked[1].CombinedScore || ranked[1].CombinedScore <= ranked[2].CombinedScore {
			t.Fatalf("unexpected calibrated ranking scores: %v, %v, %v", ranked[0].CombinedScore, ranked[1].CombinedScore, ranked[2].CombinedScore)
		}
		for _, item := range ranked {
			var reference *Result
			for _, prior := range legacy {
				if prior.ID == item.ID {
					reference = prior
					break
				}
			}
			if reference == nil || item.EmbeddingScore != reference.EmbeddingScore || item.KeywordScore != reference.KeywordScore || item.FilenameScore != reference.FilenameScore || item.CombinedScore != reference.CombinedScore {
				t.Fatalf("scoring changed for %q: got %+v, reference %+v", item.ID, item, reference)
			}
		}
	}
}

package retrieval

import (
	"math"
	"testing"
)

func TestScoreBatchKeepsInputOrderAndRerankDiagnostics(t *testing.T) {
	t.Parallel()
	weights := DefaultWeights()
	weights.Recency = 0.2
	newItems := func() []*Result {
		return []*Result{
			{ID: "weak", Content: "unrelated", Filename: "other", Embedding: []float64{0, 1}, RecencyScore: 0.5},
			{ID: "strong", Content: "alpha", Filename: "alpha", Embedding: []float64{1, 0}, RecencyScore: 0.25},
		}
	}
	batch := newItems()
	got := NewScorer(weights).ScoreBatch([]float64{1, 0}, "alpha", batch)
	ranked, traces := NewScorer(weights).RerankWithTrace([]float64{1, 0}, "alpha", newItems())
	if got[0] != batch[0] || got[0].ID != "weak" || got[1].ID != "strong" || ranked[0].ID != "strong" {
		t.Fatalf("ScoreBatch changed order: %v; rerank: %v", got, ranked)
	}
	for _, item := range got {
		for i, other := range ranked {
			if item.ID != other.ID {
				continue
			}
			if item.EmbeddingScore != other.EmbeddingScore || item.KeywordScore != other.KeywordScore || item.FilenameScore != other.FilenameScore || item.RecencyScore != other.RecencyScore || math.Abs(item.CombinedScore-traces[i].CombinedScore) > 1e-12 {
				t.Fatalf("%s diagnostics differ: batch %+v; rerank %+v", item.ID, item, other)
			}
		}
	}
}

func TestScoreBatchEmptyQueryLeavesTextSignalsAbsent(t *testing.T) {
	t.Parallel()
	items := []*Result{{Content: "alpha", Filename: "alpha.txt", KeywordScore: 1, FilenameScore: 1}}
	NewScorer(DefaultWeights()).ScoreBatch(nil, "the and", items)
	if items[0].KeywordScore != 0 || items[0].FilenameScore != 0 || items[0].CombinedScore != 0 {
		t.Fatalf("empty query scored text: %+v", items[0])
	}
}

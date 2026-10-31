package reliquary_test

import (
	"context"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestRRFUnexplainedKeepsExplainedDiagnosticsAndFusedOrder(t *testing.T) {
	t.Parallel()
	rows := func() [][]*retrieval.Result {
		return [][]*retrieval.Result{
			{{ID: "a", Content: "other", Filename: "other", Embedding: []float64{0, 1}}, {ID: "b", Content: "query", Filename: "query", Embedding: []float64{1, 0}}},
			{{ID: "a"}, {ID: "b"}},
		}
	}
	plain, err := newRRFApp(t, &rrfIndex{rows: rows()}).Search(context.Background(), "query", reliquary.WithRRF(60))
	if err != nil {
		t.Fatal(err)
	}
	explained, err := newRRFApp(t, &rrfIndex{rows: rows()}).Search(context.Background(), "query", reliquary.WithRRF(60), reliquary.WithExplain())
	if err != nil {
		t.Fatal(err)
	}
	if len(plain) != len(explained) || len(plain) != 2 {
		t.Fatalf("result lengths = %d, %d", len(plain), len(explained))
	}
	for i, item := range plain {
		other := explained[i]
		if item.ID != other.ID || item.CombinedScore != other.CombinedScore || item.EmbeddingScore != other.EmbeddingScore || item.KeywordScore != other.KeywordScore || item.FilenameScore != other.FilenameScore || item.Explain != nil || other.Explain == nil || other.Explain.HybridScoreUsed {
			t.Fatalf("result %d differs: plain %+v, explained %+v", i, item, other)
		}
	}
}

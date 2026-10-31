package reliquary_test

import (
	"context"
	"math"
	"testing"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/retrieval"
)

func TestWithExplainRRFUsesFirstDuplicateLaneRank(t *testing.T) {
	t.Parallel()
	idx := &rrfIndex{rows: [][]*retrieval.Result{
		{{ID: "a"}, {ID: "a"}, {ID: "b"}},
		{{ID: "b"}, {ID: "a"}, {ID: "a"}, {ID: "c"}},
	}}
	hits, err := newRRFApp(t, idx).Search(context.Background(), "query",
		reliquary.WithRRF(60), reliquary.WithExplain())
	if err != nil {
		t.Fatal(err)
	}
	wantRanks := map[string][2]int{"a": {1, 2}, "b": {3, 1}, "c": {0, 4}}
	if len(hits) != len(wantRanks) {
		t.Fatalf("got %d results, want %d", len(hits), len(wantRanks))
	}
	for _, hit := range hits {
		want, ok := wantRanks[hit.ID]
		if !ok {
			t.Fatalf("unexpected result %q", hit.ID)
		}
		trace := hit.Explain.RRF
		if trace.VectorRank != want[0] || trace.LexicalRank != want[1] {
			t.Fatalf("%s ranks = %d/%d, want %v", hit.ID, trace.VectorRank, trace.LexicalRank, want)
		}
		for lane, got := range []float64{trace.VectorContribution, trace.LexicalContribution} {
			wantContribution := 0.0
			if want[lane] != 0 {
				wantContribution = 30.5 / (60 + float64(want[lane]))
			}
			if math.Abs(got-wantContribution) > 1e-12 {
				t.Fatalf("%s lane %d contribution = %v, want %v", hit.ID, lane, got, wantContribution)
			}
		}
	}
}

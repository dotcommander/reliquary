package lexical

import "testing"

func TestEvaluateFixture_UsesRetrievalMetrics(t *testing.T) {
	t.Parallel()

	fixture := Fixture{
		ID:    "q1",
		Query: "alpha",
		Judgments: []Judgment{
			{ID: "doc-a", Relevance: 2, Topic: "alpha"},
			{ID: "doc-c", Relevance: 1, Topic: "gamma"},
		},
	}
	candidates := RankedList{
		{ID: "doc-b", Score: 3},
		{ID: "doc-a", Score: 2},
		{ID: "doc-c", Score: 1},
	}

	got := EvaluateFixture(fixture, candidates, 2)
	if got.RecallAtK != 0.5 {
		t.Fatalf("RecallAtK = %f, want 0.5", got.RecallAtK)
	}
	if got.PrecisionAtK != 0.5 {
		t.Fatalf("PrecisionAtK = %f, want 0.5", got.PrecisionAtK)
	}
	if got.MRR != 0.5 {
		t.Fatalf("MRR = %f, want 0.5", got.MRR)
	}
}

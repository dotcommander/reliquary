package retrieval

import "testing"

func TestEvaluatePlanReportsSourceAndLayerMetrics(t *testing.T) {
	t.Parallel()

	query := EvalQuery{
		ID:       "q1",
		Relevant: map[string]float64{"a": 1, "c": 1},
	}
	plan := Plan{
		ID:     "hybrid",
		Fusion: FusionModeRRF,
		Sources: []CandidateSource{
			{ID: "lexical", ScoreSpace: "local_bm25", Limit: 2},
			{ID: "vector", ScoreSpace: "cosine", Limit: 2},
		},
	}
	sources := []SourceReport{
		{Source: plan.Sources[0], Results: []RankedResult{{ID: "a", Score: 3}, {ID: "b", Score: 2}}},
		{Source: plan.Sources[1], Results: []RankedResult{{ID: "c", Score: 0.9}, {ID: "d", Score: 0.7}}},
	}
	layers := LayeredResults{
		Candidates: []RankedResult{{ID: "a"}, {ID: "b"}, {ID: "c"}, {ID: "d"}},
		Reranked:   []RankedResult{{ID: "c"}, {ID: "a"}},
		Final:      []RankedResult{{ID: "c"}, {ID: "a"}},
	}

	got := EvaluatePlan(query, plan, layers, sources, 2)
	if got.QueryID != "q1" || got.Plan.ID != "hybrid" {
		t.Fatalf("plan run identity = %#v", got)
	}
	if got.Report.CandidateRecall != 1 {
		t.Fatalf("candidate recall = %v, want 1", got.Report.CandidateRecall)
	}
	if len(got.Sources) != 2 {
		t.Fatalf("sources = %d, want 2", len(got.Sources))
	}
	for _, source := range got.Sources {
		if source.CandidateCount != 2 {
			t.Fatalf("%s candidate count = %d, want 2", source.Source.ID, source.CandidateCount)
		}
		if source.HitCount != 1 || source.CandidateRecall != 0.5 {
			t.Fatalf("%s hits/recall = %d/%v, want 1/0.5", source.Source.ID, source.HitCount, source.CandidateRecall)
		}
	}
}

func TestEvaluateSourceCanonicalizesReportedResults(t *testing.T) {
	t.Parallel()

	query := EvalQuery{ID: "q", Relevant: map[string]float64{"a": 1, "b": 1}}
	report := EvaluateSource(query, SourceReport{Results: []RankedResult{
		{ID: "a", Score: 3},
		{ID: "a", Score: 2},
		{ID: "b", Score: 1},
	}}, 2)
	if report.CandidateCount != 2 || report.HitCount != 2 || len(report.Results) != 2 {
		t.Fatalf("source report = %+v, want two canonical candidates and hits", report)
	}
	if report.Results[0].ID != "a" || report.Results[0].Score != 3 || report.Results[1].ID != "b" {
		t.Fatalf("reported results = %+v, want first a occurrence followed by b", report.Results)
	}
	approxEqual(t, "CandidateRecall", report.CandidateRecall, 1)
}

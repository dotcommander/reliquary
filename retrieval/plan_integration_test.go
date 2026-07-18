package retrieval

import "testing"

func TestEvaluatePlanLocalizesHybridRecallLoss(t *testing.T) {
	t.Parallel()

	query := EvalQuery{
		ID:       "hybrid-q1",
		Relevant: map[string]float64{"doc-a": 2, "doc-c": 1, "doc-e": 1},
		TopicByDoc: map[string]string{
			"doc-a": "alpha",
			"doc-c": "charlie",
			"doc-e": "echo",
		},
	}
	plan := Plan{
		ID:     "lexical-vector",
		Fusion: FusionModeRRF,
		Sources: []CandidateSource{
			{ID: "lexical", ScoreSpace: "local_bm25", Limit: 3},
			{ID: "vector", ScoreSpace: "cosine", Limit: 3},
		},
		Budgets: []StageBudget{
			{Stage: "candidates", Limit: 6},
			{Stage: "final", Limit: 3},
		},
	}
	sources := []SourceReport{
		{
			Source:  plan.Sources[0],
			Results: []RankedResult{{ID: "doc-a", Score: 9}, {ID: "doc-b", Score: 7}, {ID: "doc-c", Score: 5}},
		},
		{
			Source:  plan.Sources[1],
			Results: []RankedResult{{ID: "doc-a", Score: 0.96}, {ID: "doc-d", Score: 0.91}, {ID: "doc-f", Score: 0.88}},
		},
	}
	layers := LayeredResults{
		Candidates: []RankedResult{
			{ID: "doc-a", Score: 1},
			{ID: "doc-b", Score: 0.8},
			{ID: "doc-c", Score: 0.7},
			{ID: "doc-d", Score: 0.6},
			{ID: "doc-f", Score: 0.5},
		},
		Reranked: []RankedResult{
			{ID: "doc-a", Score: 0.99},
			{ID: "doc-d", Score: 0.83},
			{ID: "doc-b", Score: 0.8},
		},
		Final: []RankedResult{
			{ID: "doc-a", Score: 0.99},
			{ID: "doc-d", Score: 0.83},
			{ID: "doc-b", Score: 0.8},
		},
	}

	run := EvaluatePlan(query, plan, layers, sources, 3)
	if run.Report.CandidateHitCount != 2 || run.Report.CandidateRecall != float64(2)/3 {
		t.Fatalf("candidate hits/recall = %d/%v, want 2/2/3", run.Report.CandidateHitCount, run.Report.CandidateRecall)
	}
	if run.Report.FinalMetrics.RecallAtK != float64(1)/3 {
		t.Fatalf("final recall = %v, want 1/3", run.Report.FinalMetrics.RecallAtK)
	}
	if len(run.Sources) != 2 {
		t.Fatalf("source reports = %d, want 2", len(run.Sources))
	}
	if run.Sources[0].CandidateRecall != float64(2)/3 {
		t.Fatalf("lexical recall = %v, want 2/3", run.Sources[0].CandidateRecall)
	}
	if run.Sources[1].CandidateRecall != float64(1)/3 {
		t.Fatalf("vector recall = %v, want 1/3", run.Sources[1].CandidateRecall)
	}
}

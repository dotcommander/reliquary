package retrieval

import "testing"

func TestTuneWeightsRejectsInvalidConfigsAndSelectsBest(t *testing.T) {
	t.Parallel()

	cases := []TuneCase{
		{
			Query: EvalQuery{ID: "q1", Relevant: map[string]float64{"rel": 1}},
			Candidates: []TuneCandidate{
				{ID: "rel", Signals: ScoreSignals{Embedding: 1, Keyword: 0}, Topic: "topic-rel"},
				{ID: "bad", Signals: ScoreSignals{Embedding: 0, Keyword: 1}, Topic: "topic-bad"},
			},
		},
	}
	config := TuneConfig{
		K: 1,
		Weights: []Weights{
			{Keyword: 1},
			{Embedding: 1},
		},
		Constraints: TuneConstraints{MinRecallAtK: 1, MinNDCGAtK: 1},
	}

	report := TuneWeights(cases, config)
	if len(report.Results) != 2 {
		t.Fatalf("results len = %d, want 2", len(report.Results))
	}
	if !report.Results[0].Rejected || report.Results[0].RejectReason != "recall_at_k" {
		t.Fatalf("first result rejection = %v/%q, want recall_at_k", report.Results[0].Rejected, report.Results[0].RejectReason)
	}
	if report.Results[1].Rejected {
		t.Fatalf("second result rejected = true, want false: %+v", report.Results[1])
	}
	if !report.HasBest {
		t.Fatal("HasBest = false, want true")
	}
	if report.Best.Weights.Embedding != 1 {
		t.Fatalf("best weights = %+v, want embedding config", report.Best.Weights)
	}
	approxEqual(t, "best RecallAtK", report.Best.Metrics.RecallAtK, 1)
	approxEqual(t, "best NDCGAtK", report.Best.Metrics.NDCGAtK, 1)
}

func TestTuneWeightsTieBreaksDeterministically(t *testing.T) {
	t.Parallel()

	cases := []TuneCase{
		{
			Query: EvalQuery{ID: "q1", Relevant: map[string]float64{"a": 1}},
			Candidates: []TuneCandidate{
				{ID: "a", Signals: ScoreSignals{Embedding: 1, Keyword: 1}},
				{ID: "b", Signals: ScoreSignals{Embedding: 0, Keyword: 0}},
			},
		},
	}
	report := TuneWeights(cases, TuneConfig{
		K: 1,
		Weights: []Weights{
			{Embedding: 1},
			{Keyword: 1},
		},
	})
	if !report.HasBest {
		t.Fatal("HasBest = false, want true")
	}
	if report.Best.Weights.Keyword != 1 || report.Best.Weights.Embedding != 0 {
		t.Fatalf("best weights = %+v, want lexicographically smaller keyword config", report.Best.Weights)
	}
}

func TestTuneWeightsMMRLambdaAndUniqueTopicConstraint(t *testing.T) {
	t.Parallel()

	cases := []TuneCase{
		{
			Query: EvalQuery{
				ID:         "q1",
				Relevant:   map[string]float64{"a": 1, "b": 1},
				TopicByDoc: map[string]string{"a": "alpha", "b": "beta", "a2": "alpha"},
			},
			Candidates: []TuneCandidate{
				{ID: "a", Signals: ScoreSignals{Embedding: 1}, Embedding: []float64{1, 0}, Topic: "alpha"},
				{ID: "a2", Signals: ScoreSignals{Embedding: 0.99}, Embedding: []float64{1, 0}, Topic: "alpha"},
				{ID: "b", Signals: ScoreSignals{Embedding: 0.7}, Embedding: []float64{0, 1}, Topic: "beta"},
			},
		},
	}
	report := TuneWeights(cases, TuneConfig{
		K:           2,
		Weights:     []Weights{{Embedding: 1}},
		MMRLambdas:  []float64{1, 0},
		Constraints: TuneConstraints{MinUniqueTopicsAtK: 2},
	})
	if len(report.Results) != 2 {
		t.Fatalf("results len = %d, want 2", len(report.Results))
	}
	if !report.Results[0].Rejected || report.Results[0].RejectReason != "unique_topics_at_k" {
		t.Fatalf("lambda=1 rejection = %v/%q, want unique topic rejection", report.Results[0].Rejected, report.Results[0].RejectReason)
	}
	if report.Results[1].Rejected {
		t.Fatalf("lambda=0 rejected = true, want false: %+v", report.Results[1])
	}
	if !report.Best.UsedMMR || report.Best.MMRLambda != 0 {
		t.Fatalf("best MMR = used %v lambda %v, want used true lambda 0", report.Best.UsedMMR, report.Best.MMRLambda)
	}
}

func TestTuneWeightsInvalidInput(t *testing.T) {
	t.Parallel()

	if got := TuneWeights(nil, TuneConfig{K: 1, Weights: []Weights{{Embedding: 1}}}); len(got.Results) != 0 || got.HasBest {
		t.Fatalf("nil cases report = %+v, want empty", got)
	}
	if got := TuneWeights([]TuneCase{{}}, TuneConfig{K: 0, Weights: []Weights{{Embedding: 1}}}); len(got.Results) != 0 || got.HasBest {
		t.Fatalf("k=0 report = %+v, want empty", got)
	}
	if got := TuneWeights([]TuneCase{{}}, TuneConfig{K: 1}); len(got.Results) != 0 || got.HasBest {
		t.Fatalf("empty weights report = %+v, want empty", got)
	}
}

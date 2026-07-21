package retrieval

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestEvaluateRunReportsFixtureMetrics(t *testing.T) {
	t.Parallel()

	fixture := Fixture{
		ID: "golden",
		Queries: []FixtureQuery{
			{
				ID:   "q2",
				Text: "go contexts",
				Judgments: []Judgment{
					{DocID: "doc-d", Relevance: 1, Topic: "go", Segment: "beta"},
				},
			},
			{
				ID:   "q1",
				Text: "ranking metrics",
				Judgments: []Judgment{
					{DocID: "doc-a", Relevance: 2, Topic: "ml", Segment: "alpha"},
					{DocID: "doc-b", Relevance: 1, Topic: "systems", Segment: "beta"},
					{DocID: "doc-c", Relevance: 0, Topic: "noise", Segment: "alpha"},
				},
			},
		},
	}
	run := Run{
		ID: "candidate",
		Queries: []RunQuery{
			{
				ID: "q2",
				Results: []RankedResult{
					{ID: "doc-z", Score: 0.9},
					{ID: "doc-d", Score: 0.8},
				},
			},
			{
				ID: "q1",
				Results: []RankedResult{
					{ID: "doc-b", Score: 0.9},
					{ID: "doc-c", Score: 0.8},
					{ID: "doc-a", Score: 0.7},
				},
				Stages: StageResults{
					Candidates: []RankedResult{
						{ID: "doc-a", Score: 0.3},
						{ID: "doc-b", Score: 0.2},
					},
					Reranked: []RankedResult{
						{ID: "doc-c", Score: 0.9},
						{ID: "doc-b", Score: 0.8},
					},
					Diversified: []RankedResult{
						{ID: "doc-b", Score: 0.8},
						{ID: "doc-a", Score: 0.7},
					},
					Final: []RankedResult{
						{ID: "doc-b", Score: 0.8},
					},
				},
			},
		},
	}

	report, err := EvaluateRun(fixture, run, 2)
	if err != nil {
		t.Fatalf("EvaluateRun returned error: %v", err)
	}
	if report.FixtureID != "golden" || report.RunID != "candidate" || report.K != 2 || report.QueryCount != 2 {
		t.Fatalf("report identity/count fields = %+v, want fixture/run/k/query count", report)
	}
	if len(report.Queries) != 2 || report.Queries[0].ID != "q1" || report.Queries[1].ID != "q2" {
		t.Fatalf("query order = %+v, want lexical q1,q2", report.Queries)
	}

	log2_3 := math.Log2(3)
	q1 := report.Queries[0]
	approxEqual(t, "q1 RecallAtK", q1.Metrics.RecallAtK, 0.5)
	approxEqual(t, "q1 PrecisionAtK", q1.Metrics.PrecisionAtK, 0.5)
	approxEqual(t, "q1 MRR", q1.Metrics.MRR, 1)
	approxEqual(t, "q1 NDCGAtK", q1.Metrics.NDCGAtK, 1/(3+1/log2_3))
	if q1.Metrics.UniqueTopicAtK != 2 {
		t.Fatalf("q1 UniqueTopicAtK = %d, want 2", q1.Metrics.UniqueTopicAtK)
	}

	q2 := report.Queries[1]
	approxEqual(t, "q2 RecallAtK", q2.Metrics.RecallAtK, 1)
	approxEqual(t, "q2 PrecisionAtK", q2.Metrics.PrecisionAtK, 0.5)
	approxEqual(t, "q2 MRR", q2.Metrics.MRR, 0.5)
	approxEqual(t, "q2 NDCGAtK", q2.Metrics.NDCGAtK, 1/log2_3)
	if q2.Metrics.UniqueTopicAtK != 1 {
		t.Fatalf("q2 UniqueTopicAtK = %d, want 1", q2.Metrics.UniqueTopicAtK)
	}

	approxEqual(t, "aggregate RecallAtK", report.Metrics.RecallAtK, 0.75)
	approxEqual(t, "aggregate PrecisionAtK", report.Metrics.PrecisionAtK, 0.5)
	approxEqual(t, "aggregate MRR", report.Metrics.MRR, 0.75)
	approxEqual(t, "aggregate NDCGAtK", report.Metrics.NDCGAtK, (1/(3+1/log2_3)+1/log2_3)/2)
	if report.Metrics.UniqueTopicAtK != 1 {
		t.Fatalf("aggregate UniqueTopicAtK = %d, want integer average 1", report.Metrics.UniqueTopicAtK)
	}
}

func TestValidateFixtureRejectsInvalidInput(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		fixture Fixture
		want    string
	}{
		{
			name:    "missing fixture ID",
			fixture: Fixture{Queries: []FixtureQuery{{ID: "q", Judgments: []Judgment{{DocID: "d"}}}}},
			want:    "fixture ID",
		},
		{
			name:    "missing query ID",
			fixture: Fixture{ID: "f", Queries: []FixtureQuery{{Judgments: []Judgment{{DocID: "d"}}}}},
			want:    "ID is required",
		},
		{
			name: "duplicate query ID",
			fixture: Fixture{ID: "f", Queries: []FixtureQuery{
				{ID: "q", Judgments: []Judgment{{DocID: "a"}}},
				{ID: "q", Judgments: []Judgment{{DocID: "b"}}},
			}},
			want: "duplicate query ID",
		},
		{
			name:    "empty judgments",
			fixture: Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q"}}},
			want:    "at least one judgment",
		},
		{
			name:    "missing doc ID",
			fixture: Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q", Judgments: []Judgment{{}}}}},
			want:    "doc ID",
		},
		{
			name:    "duplicate doc ID",
			fixture: Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q", Judgments: []Judgment{{DocID: "d"}, {DocID: "d"}}}}},
			want:    "duplicate doc ID",
		},
		{
			name:    "negative relevance",
			fixture: Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q", Judgments: []Judgment{{DocID: "d", Relevance: -1}}}}},
			want:    "non-negative",
		},
		{
			name:    "nan relevance",
			fixture: Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q", Judgments: []Judgment{{DocID: "d", Relevance: math.NaN()}}}}},
			want:    "finite",
		},
		{
			name:    "infinite relevance",
			fixture: Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q", Judgments: []Judgment{{DocID: "d", Relevance: math.Inf(1)}}}}},
			want:    "finite",
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateFixture(tc.fixture)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidateFixture error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestValidateRunRejectsInvalidInput(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		run  Run
		want string
	}{
		{name: "missing run ID", run: Run{Queries: []RunQuery{{ID: "q", Results: []RankedResult{{ID: "d"}}}}}, want: "run ID"},
		{name: "missing query ID", run: Run{ID: "r", Queries: []RunQuery{{Results: []RankedResult{{ID: "d"}}}}}, want: "ID is required"},
		{name: "duplicate query ID", run: Run{ID: "r", Queries: []RunQuery{{ID: "q", Results: []RankedResult{{ID: "a"}}}, {ID: "q", Results: []RankedResult{{ID: "b"}}}}}, want: "duplicate query ID"},
		{name: "missing results and stages", run: Run{ID: "r", Queries: []RunQuery{{ID: "q"}}}, want: "results or stage results"},
		{name: "nan result score", run: Run{ID: "r", Queries: []RunQuery{{ID: "q", Results: []RankedResult{{ID: "d", Score: math.NaN()}}}}}, want: "finite"},
		{name: "infinite stage score", run: Run{ID: "r", Queries: []RunQuery{{ID: "q", Stages: StageResults{Final: []RankedResult{{ID: "d", Score: math.Inf(1)}}}}}}, want: "finite"},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateRun(tc.run)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidateRun error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestValidateRunRejectsBlankAndDuplicateIDsInEveryResultList(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		set  func(*RunQuery, []RankedResult)
	}{
		{name: "results", set: func(query *RunQuery, results []RankedResult) { query.Results = results }},
		{name: "candidates", set: func(query *RunQuery, results []RankedResult) { query.Stages.Candidates = results }},
		{name: "reranked", set: func(query *RunQuery, results []RankedResult) { query.Stages.Reranked = results }},
		{name: "diversified", set: func(query *RunQuery, results []RankedResult) { query.Stages.Diversified = results }},
		{name: "final", set: func(query *RunQuery, results []RankedResult) { query.Stages.Final = results }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			for _, tc := range []struct {
				name    string
				results []RankedResult
				want    string
			}{
				{name: "blank", results: []RankedResult{{ID: " \t"}}, want: "ID is required"},
				{name: "duplicate", results: []RankedResult{{ID: "a"}, {ID: "a"}}, want: "duplicate ID"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					t.Parallel()
					query := RunQuery{ID: "q"}
					tt.set(&query, tc.results)
					err := ValidateRun(Run{ID: "r", Queries: []RunQuery{query}})
					if err == nil || !strings.Contains(err.Error(), tc.want) {
						t.Fatalf("ValidateRun error = %v, want substring %q", err, tc.want)
					}
				})
			}
		})
	}
}

func TestEvaluateRunErrorsOnMissingFixtureQuery(t *testing.T) {
	t.Parallel()

	_, err := EvaluateRun(
		Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q1", Judgments: []Judgment{{DocID: "d", Relevance: 1}}}}},
		Run{ID: "r", Queries: []RunQuery{{ID: "missing", Results: []RankedResult{{ID: "d"}}}}},
		1,
	)
	if err == nil || !strings.Contains(err.Error(), "fixture query not found") {
		t.Fatalf("EvaluateRun error = %v, want missing fixture query", err)
	}
}

func TestEvaluateRunRequiresCompleteFixtureCoverage(t *testing.T) {
	t.Parallel()

	fixture := Fixture{ID: "f", Queries: []FixtureQuery{
		{ID: "q3", Judgments: []Judgment{{DocID: "hard", Relevance: 1}}},
		{ID: "q1", Judgments: []Judgment{{DocID: "easy", Relevance: 1}}},
		{ID: "q2", Judgments: []Judgment{{DocID: "medium", Relevance: 1}}},
	}}
	tests := []struct {
		name string
		run  Run
		want string
	}{
		{
			name: "partial run omits hard query",
			run:  Run{ID: "r", Queries: []RunQuery{{ID: "q1", Results: []RankedResult{{ID: "easy"}}}, {ID: "q2", Results: []RankedResult{{ID: "medium"}}}}},
			want: "run is missing fixture queries: q3",
		},
		{
			name: "empty run omits every query in lexical order",
			run:  Run{ID: "r"},
			want: "run is missing fixture queries: q1, q2, q3",
		},
		{
			name: "multiple missing queries are ordered lexically",
			run:  Run{ID: "r", Queries: []RunQuery{{ID: "q2", Results: []RankedResult{{ID: "medium"}}}}},
			want: "run is missing fixture queries: q1, q3",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := EvaluateRun(fixture, tc.run, 1)
			if err == nil || err.Error() != tc.want {
				t.Fatalf("EvaluateRun error = %v, want %q", err, tc.want)
			}
		})
	}

	report, err := EvaluateRun(Fixture{ID: "empty"}, Run{ID: "empty"}, 1)
	if err != nil {
		t.Fatalf("EvaluateRun empty fixture and run returned error: %v", err)
	}
	if report.QueryCount != 0 {
		t.Fatalf("EvaluateRun empty fixture and run query count = %d, want 0", report.QueryCount)
	}
}

func TestEvaluateRunSegmentsFromJudgments(t *testing.T) {
	t.Parallel()

	report, err := EvaluateRun(
		Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q", Judgments: []Judgment{
			{DocID: "b-rel", Relevance: 1, Segment: "beta"},
			{DocID: "a-rel", Relevance: 1, Segment: "alpha"},
			{DocID: "a-noise", Relevance: 0, Segment: "alpha"},
		}}}},
		Run{ID: "r", Queries: []RunQuery{{ID: "q", Results: []RankedResult{
			{ID: "b-rel", Score: 0.9},
			{ID: "a-noise", Score: 0.8},
			{ID: "a-rel", Score: 0.7},
		}}}},
		2,
	)
	if err != nil {
		t.Fatalf("EvaluateRun returned error: %v", err)
	}
	segments := report.Queries[0].Segments
	if len(segments) != 2 || segments[0].Segment != "alpha" || segments[1].Segment != "beta" {
		t.Fatalf("segments = %+v, want alpha,beta lexical order", segments)
	}
	if segments[0].RelevantCount != 1 || segments[0].ResultCount != 1 || segments[0].HitCount != 0 {
		t.Fatalf("alpha segment = %+v, want relevant=1 result=1 hit=0", segments[0])
	}
	if segments[1].RelevantCount != 1 || segments[1].ResultCount != 1 || segments[1].HitCount != 1 {
		t.Fatalf("beta segment = %+v, want relevant=1 result=1 hit=1", segments[1])
	}
}

func TestEvaluateRunLayersUseCapturedStages(t *testing.T) {
	t.Parallel()

	report, err := EvaluateRun(
		Fixture{ID: "f", Queries: []FixtureQuery{{ID: "q", Judgments: []Judgment{
			{DocID: "rel-a", Relevance: 1, Topic: "alpha"},
			{DocID: "rel-b", Relevance: 1, Topic: "beta"},
		}}}},
		Run{ID: "r", Queries: []RunQuery{{ID: "q", Stages: StageResults{
			Candidates:  []RankedResult{{ID: "rel-a"}, {ID: "noise"}, {ID: "rel-b"}},
			Reranked:    []RankedResult{{ID: "noise", Topic: "alpha"}, {ID: "rel-a"}},
			Diversified: []RankedResult{{ID: "rel-a"}, {ID: "rel-b"}},
			Final:       []RankedResult{{ID: "rel-b"}},
		}}}},
		2,
	)
	if err != nil {
		t.Fatalf("EvaluateRun returned error: %v", err)
	}
	layers := report.Queries[0].Layers
	if layers.CandidateCount != 3 || layers.CandidateHitCount != 2 {
		t.Fatalf("layer candidate counts = %+v, want 3 candidates and 2 hits", layers)
	}
	approxEqual(t, "CandidateRecall", layers.CandidateRecall, 1)
	approxEqual(t, "RerankMetrics MRR", layers.RerankMetrics.MRR, 0.5)
	approxEqual(t, "DiversifiedMetrics RecallAtK", layers.DiversifiedMetrics.RecallAtK, 1)
	approxEqual(t, "FinalMetrics RecallAtK", layers.FinalMetrics.RecallAtK, 0.5)
	if layers.DiversityLiftAtK != 1 {
		t.Fatalf("DiversityLiftAtK = %d, want 1", layers.DiversityLiftAtK)
	}
}

func TestCheckThresholds(t *testing.T) {
	t.Parallel()

	report := Report{Metrics: Metrics{
		RecallAtK:      0.8,
		PrecisionAtK:   0.6,
		MRR:            0.7,
		NDCGAtK:        0.9,
		UniqueTopicAtK: 2,
	}}
	passing := ReportThresholds{
		MinRecallAtK:       0.8,
		MinPrecisionAtK:    0.6,
		MinMRR:             0.7,
		MinNDCGAtK:         0.9,
		MinUniqueTopicsAtK: 2,
	}
	if failures := CheckThresholds(report, passing); len(failures) != 0 {
		t.Fatalf("passing thresholds failures = %+v, want none", failures)
	}

	failing := ReportThresholds{
		MinRecallAtK:       0.81,
		MinPrecisionAtK:    0.61,
		MinMRR:             0.71,
		MinNDCGAtK:         0.91,
		MinUniqueTopicsAtK: 3,
	}
	failures := CheckThresholds(report, failing)
	gotNames := make([]string, 0, len(failures))
	for _, failure := range failures {
		gotNames = append(gotNames, failure.Metric)
	}
	wantNames := []string{"recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "unique_topics_at_k"}
	if !reflect.DeepEqual(gotNames, wantNames) {
		t.Fatalf("failure names = %v, want %v; failures = %+v", gotNames, wantNames, failures)
	}
	if failures[4].Got != 2 || failures[4].Want != 3 {
		t.Fatalf("unique topic failure = %+v, want got/want 2/3", failures[4])
	}
}

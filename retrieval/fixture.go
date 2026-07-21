package retrieval

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// Fixture is a caller-owned golden query set for deterministic retrieval
// quality reports.
type Fixture struct {
	ID          string
	Description string
	Queries     []FixtureQuery
}

// FixtureQuery is one golden query and its judged documents.
type FixtureQuery struct {
	ID        string
	Text      string
	Judgments []Judgment
}

// Judgment labels one document for a fixture query. Relevance values greater
// than zero are relevant; zero relevance can still provide topic or segment
// metadata for evaluated results.
type Judgment struct {
	DocID     string
	Relevance float64
	Topic     string
	Segment   string
}

// Run is a captured retrieval run to evaluate against a Fixture.
type Run struct {
	ID      string
	Queries []RunQuery
}

// RunQuery contains the primary ranked results and optional stage outputs for
// one fixture query.
type RunQuery struct {
	ID      string
	Results []RankedResult
	Stages  StageResults
}

// StageResults aliases the existing layered retrieval stage shape.
type StageResults = LayeredResults

// Report is the aggregate and per-query fixture evaluation output.
type Report struct {
	FixtureID  string
	RunID      string
	K          int
	QueryCount int
	Metrics    Metrics
	Queries    []ReportQuery
}

// ReportQuery is the per-query portion of a fixture evaluation report.
type ReportQuery struct {
	ID       string
	Metrics  Metrics
	Segments []SegmentMetrics
	Layers   LayerReport
}

// ReportThresholds declares aggregate metric floors for CheckThresholds.
type ReportThresholds struct {
	MinRecallAtK       float64
	MinPrecisionAtK    float64
	MinMRR             float64
	MinNDCGAtK         float64
	MinUniqueTopicsAtK int
}

// ThresholdFailure reports one aggregate metric below its configured floor.
type ThresholdFailure struct {
	Metric string
	Got    float64
	Want   float64
}

// ValidateFixture checks the structural invariants required for fixture
// evaluation.
func ValidateFixture(f Fixture) error {
	if f.ID == "" {
		return fmt.Errorf("fixture ID is required")
	}
	queryIDs := make(map[string]struct{}, len(f.Queries))
	for i, query := range f.Queries {
		if query.ID == "" {
			return fmt.Errorf("fixture query at index %d: ID is required", i)
		}
		if _, exists := queryIDs[query.ID]; exists {
			return fmt.Errorf("fixture query %q: duplicate query ID", query.ID)
		}
		queryIDs[query.ID] = struct{}{}
		if len(query.Judgments) == 0 {
			return fmt.Errorf("fixture query %q: at least one judgment is required", query.ID)
		}
		docIDs := make(map[string]struct{}, len(query.Judgments))
		for j, judgment := range query.Judgments {
			if judgment.DocID == "" {
				return fmt.Errorf("fixture query %q judgment at index %d: doc ID is required", query.ID, j)
			}
			if _, exists := docIDs[judgment.DocID]; exists {
				return fmt.Errorf("fixture query %q judgment %q: duplicate doc ID", query.ID, judgment.DocID)
			}
			docIDs[judgment.DocID] = struct{}{}
			if judgment.Relevance < 0 {
				return fmt.Errorf("fixture query %q judgment %q: relevance must be non-negative", query.ID, judgment.DocID)
			}
			if math.IsNaN(judgment.Relevance) || math.IsInf(judgment.Relevance, 0) {
				return fmt.Errorf("fixture query %q judgment %q: relevance must be finite", query.ID, judgment.DocID)
			}
		}
	}
	return nil
}

// ValidateRun checks the structural invariants required for run evaluation.
func ValidateRun(r Run) error {
	if r.ID == "" {
		return fmt.Errorf("run ID is required")
	}
	queryIDs := make(map[string]struct{}, len(r.Queries))
	for i, query := range r.Queries {
		if query.ID == "" {
			return fmt.Errorf("run query at index %d: ID is required", i)
		}
		if _, exists := queryIDs[query.ID]; exists {
			return fmt.Errorf("run query %q: duplicate query ID", query.ID)
		}
		queryIDs[query.ID] = struct{}{}
		if len(query.Results) == 0 && emptyStages(query.Stages) {
			return fmt.Errorf("run query %q: results or stage results are required", query.ID)
		}
		if err := validateRankedResults(query.Results, fmt.Sprintf("run query %q results", query.ID)); err != nil {
			return err
		}
		if err := validateRankedResults(query.Stages.Candidates, fmt.Sprintf("run query %q candidates", query.ID)); err != nil {
			return err
		}
		if err := validateRankedResults(query.Stages.Reranked, fmt.Sprintf("run query %q reranked", query.ID)); err != nil {
			return err
		}
		if err := validateRankedResults(query.Stages.Diversified, fmt.Sprintf("run query %q diversified", query.ID)); err != nil {
			return err
		}
		if err := validateRankedResults(query.Stages.Final, fmt.Sprintf("run query %q final", query.ID)); err != nil {
			return err
		}
	}
	return nil
}

// EvalQuery converts a fixture query into the existing evaluation shape.
func (fq FixtureQuery) EvalQuery() EvalQuery {
	return EvalQueryFromFixture(fq)
}

// EvalQueryFromFixture converts positive judgments into Relevant and all
// judgment topics into TopicByDoc.
func EvalQueryFromFixture(fq FixtureQuery) EvalQuery {
	query := EvalQuery{
		ID:         fq.ID,
		Relevant:   make(map[string]float64),
		TopicByDoc: make(map[string]string),
	}
	for _, judgment := range fq.Judgments {
		if judgment.Relevance > 0 {
			query.Relevant[judgment.DocID] = judgment.Relevance
		}
		if judgment.Topic != "" {
			query.TopicByDoc[judgment.DocID] = judgment.Topic
		}
	}
	return query
}

// EvaluateRun evaluates captured retrieval results against a validated fixture.
// The run must contain every fixture query and no unknown query IDs. Query
// reports are sorted lexically by query ID, independent of fixture or run input
// order.
func EvaluateRun(f Fixture, r Run, k int) (Report, error) {
	if err := ValidateFixture(f); err != nil {
		return Report{}, err
	}
	if err := ValidateRun(r); err != nil {
		return Report{}, err
	}

	fixtureQueries := make(map[string]FixtureQuery, len(f.Queries))
	for _, query := range f.Queries {
		fixtureQueries[query.ID] = query
	}

	runQueries := slices.Clone(r.Queries)
	slices.SortFunc(runQueries, func(a, b RunQuery) int {
		switch {
		case a.ID < b.ID:
			return -1
		case a.ID > b.ID:
			return 1
		default:
			return 0
		}
	})
	runQueryIDs := make(map[string]struct{}, len(runQueries))
	for _, query := range runQueries {
		if _, exists := fixtureQueries[query.ID]; !exists {
			return Report{}, fmt.Errorf("run query %q: fixture query not found", query.ID)
		}
		runQueryIDs[query.ID] = struct{}{}
	}

	missingQueryIDs := make([]string, 0, len(fixtureQueries)-len(runQueryIDs))
	for queryID := range fixtureQueries {
		if _, exists := runQueryIDs[queryID]; !exists {
			missingQueryIDs = append(missingQueryIDs, queryID)
		}
	}
	if len(missingQueryIDs) > 0 {
		slices.Sort(missingQueryIDs)
		return Report{}, fmt.Errorf("run is missing fixture queries: %s", strings.Join(missingQueryIDs, ", "))
	}

	report := Report{
		FixtureID: f.ID,
		RunID:     r.ID,
		K:         k,
		Queries:   make([]ReportQuery, 0, len(runQueries)),
	}
	for _, runQuery := range runQueries {
		fixtureQuery := fixtureQueries[runQuery.ID]
		evalQuery := fixtureQuery.EvalQuery()
		metrics := Evaluate(evalQuery, runQuery.Results, k)
		report.Queries = append(report.Queries, ReportQuery{
			ID:       runQuery.ID,
			Metrics:  metrics,
			Segments: EvaluateSegments(evalQuery, runQuery.Results, k, segmenterForJudgments(fixtureQuery.Judgments)),
			Layers:   EvaluateLayers(evalQuery, runQuery.Stages, k),
		})
		report.Metrics.RecallAtK += metrics.RecallAtK
		report.Metrics.PrecisionAtK += metrics.PrecisionAtK
		report.Metrics.MRR += metrics.MRR
		report.Metrics.NDCGAtK += metrics.NDCGAtK
		report.Metrics.UniqueTopicAtK += metrics.UniqueTopicAtK
	}

	report.QueryCount = len(report.Queries)
	if report.QueryCount > 0 {
		count := float64(report.QueryCount)
		report.Metrics.RecallAtK /= count
		report.Metrics.PrecisionAtK /= count
		report.Metrics.MRR /= count
		report.Metrics.NDCGAtK /= count
		report.Metrics.UniqueTopicAtK = int(float64(report.Metrics.UniqueTopicAtK) / count)
	}
	return report, nil
}

// CheckThresholds compares aggregate report metrics to configured floors. The
// returned failures are ordered by stable metric name order.
func CheckThresholds(report Report, thresholds ReportThresholds) []ThresholdFailure {
	failures := make([]ThresholdFailure, 0, 5)
	if report.Metrics.RecallAtK < thresholds.MinRecallAtK {
		failures = append(failures, ThresholdFailure{Metric: "recall_at_k", Got: report.Metrics.RecallAtK, Want: thresholds.MinRecallAtK})
	}
	if report.Metrics.PrecisionAtK < thresholds.MinPrecisionAtK {
		failures = append(failures, ThresholdFailure{Metric: "precision_at_k", Got: report.Metrics.PrecisionAtK, Want: thresholds.MinPrecisionAtK})
	}
	if report.Metrics.MRR < thresholds.MinMRR {
		failures = append(failures, ThresholdFailure{Metric: "mrr", Got: report.Metrics.MRR, Want: thresholds.MinMRR})
	}
	if report.Metrics.NDCGAtK < thresholds.MinNDCGAtK {
		failures = append(failures, ThresholdFailure{Metric: "ndcg_at_k", Got: report.Metrics.NDCGAtK, Want: thresholds.MinNDCGAtK})
	}
	if report.Metrics.UniqueTopicAtK < thresholds.MinUniqueTopicsAtK {
		failures = append(failures, ThresholdFailure{
			Metric: "unique_topics_at_k",
			Got:    float64(report.Metrics.UniqueTopicAtK),
			Want:   float64(thresholds.MinUniqueTopicsAtK),
		})
	}
	return failures
}

func emptyStages(stages StageResults) bool {
	return len(stages.Candidates) == 0 &&
		len(stages.Reranked) == 0 &&
		len(stages.Diversified) == 0 &&
		len(stages.Final) == 0
}

func validateRankedResults(results []RankedResult, label string) error {
	ids := make(map[string]struct{}, len(results))
	for i, result := range results {
		if strings.TrimSpace(result.ID) == "" {
			return fmt.Errorf("%s result at index %d: ID is required", label, i)
		}
		if _, exists := ids[result.ID]; exists {
			return fmt.Errorf("%s result at index %d: duplicate ID %q", label, i, result.ID)
		}
		ids[result.ID] = struct{}{}
		if math.IsNaN(result.Score) || math.IsInf(result.Score, 0) {
			return fmt.Errorf("%s result at index %d: score must be finite", label, i)
		}
	}
	return nil
}

func segmenterForJudgments(judgments []Judgment) Segmenter {
	segmentByDoc := make(map[string]string, len(judgments))
	for _, judgment := range judgments {
		if judgment.Segment != "" {
			segmentByDoc[judgment.DocID] = judgment.Segment
		}
	}
	return func(docID string) string {
		return segmentByDoc[docID]
	}
}

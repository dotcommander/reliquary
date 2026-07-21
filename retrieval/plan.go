package retrieval

import "github.com/dotcommander/reliquary/internal/hash"

// FusionMode labels how source candidate lists are combined. It is a caller
// contract only; this package does not execute provider queries or rerankers.
type FusionMode string

const (
	FusionModeNone     FusionMode = ""
	FusionModeRRF      FusionMode = "rrf"
	FusionModeWeighted FusionMode = "weighted"
)

// CandidateSource describes one provider-neutral candidate source in a
// retrieval plan.
type CandidateSource struct {
	ID         string
	ScoreSpace string
	Limit      int
	Weight     float64
}

// StageBudget describes a limit for a named retrieval stage.
type StageBudget struct {
	Stage string
	Limit int
}

// Plan describes source budgets and stage labels for a retrieval run.
type Plan struct {
	ID             string
	Sources        []CandidateSource
	Fusion         FusionMode
	RerankLabel    string
	DiversifyLabel string
	Budgets        []StageBudget
	Identity       hash.Digest
}

// SourceReport captures ranked output and metrics for one candidate source.
type SourceReport struct {
	Source          CandidateSource
	Results         []RankedResult
	CandidateCount  int
	HitCount        int
	CandidateRecall float64
	Metrics         Metrics
}

// PlanRun captures the observed outputs for a retrieval plan.
type PlanRun struct {
	Plan    Plan
	QueryID string
	Sources []SourceReport
	Layers  LayeredResults
	Report  LayerReport
}

// EvaluatePlan builds a PlanRun with per-source and layered metrics.
func EvaluatePlan(query EvalQuery, plan Plan, layers LayeredResults, sources []SourceReport, k int) PlanRun {
	evaluated := make([]SourceReport, len(sources))
	for i, source := range sources {
		evaluated[i] = EvaluateSource(query, source, k)
	}
	return PlanRun{
		Plan:    plan,
		QueryID: query.ID,
		Sources: evaluated,
		Layers:  layers,
		Report:  EvaluateLayers(query, layers, k),
	}
}

// EvaluateSource fills metrics for one candidate source report.
func EvaluateSource(query EvalQuery, report SourceReport, k int) SourceReport {
	report.Results = canonicalRankedResults(report.Results)
	report.CandidateCount = len(report.Results)
	report.HitCount = hitCount(query.Relevant, report.Results)
	if len(query.Relevant) > 0 {
		report.CandidateRecall = float64(report.HitCount) / float64(len(query.Relevant))
	}
	report.Metrics = Evaluate(query, report.Results, k)
	return report
}

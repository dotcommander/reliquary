package retrieval

import (
	"cmp"
	"slices"
)

// TuneCandidate is a precomputed candidate row for retrieval tuning. Signals
// are already normalized into caller-owned comparable spaces.
type TuneCandidate struct {
	ID        string
	Signals   ScoreSignals
	Embedding []float64
	Topic     string
}

// TuneCase is one labeled retrieval query and its candidate set.
type TuneCase struct {
	Query      EvalQuery
	Candidates []TuneCandidate
}

// TuneConstraints reject grid configurations that fail required floor metrics.
type TuneConstraints struct {
	MinRecallAtK       float64
	MinNDCGAtK         float64
	MinUniqueTopicsAtK int
}

// TuneConfig configures deterministic grid search over weights and optional MMR
// lambdas. Empty MMRLambdas evaluates plain weighted ranking only.
type TuneConfig struct {
	K           int
	Weights     []Weights
	MMRLambdas  []float64
	Constraints TuneConstraints
}

// TuneResult reports one grid configuration's aggregate metrics and rejection state.
type TuneResult struct {
	Weights      Weights
	MMRLambda    float64
	UsedMMR      bool
	Metrics      Metrics
	QueryCount   int
	Rejected     bool
	RejectReason string
}

// TuneReport contains all grid results plus the best non-rejected result.
type TuneReport struct {
	Results []TuneResult
	Best    TuneResult
	HasBest bool
}

// TuneWeights evaluates each weight/lambda configuration, rejects configs below
// constraints, and selects the best remaining result by deterministic tie-breaks.
func TuneWeights(cases []TuneCase, config TuneConfig) TuneReport {
	if config.K <= 0 || len(cases) == 0 || len(config.Weights) == 0 {
		return TuneReport{}
	}
	lambdas := config.MMRLambdas
	if len(lambdas) == 0 {
		lambdas = []float64{0}
	}

	report := TuneReport{}
	for _, weights := range config.Weights {
		for _, lambda := range lambdas {
			usedMMR := len(config.MMRLambdas) > 0
			result := TuneResult{
				Weights:    weights,
				MMRLambda:  clamp01(lambda),
				UsedMMR:    usedMMR,
				QueryCount: len(cases),
				Metrics:    evaluateTuneConfig(cases, weights, config.K, lambda, usedMMR),
			}
			result.Rejected, result.RejectReason = rejectTuneResult(result.Metrics, config.Constraints)
			report.Results = append(report.Results, result)
		}
	}

	for _, result := range report.Results {
		if result.Rejected {
			continue
		}
		if !report.HasBest || tuneResultBetter(result, report.Best) {
			report.Best = result
			report.HasBest = true
		}
	}
	return report
}

func evaluateTuneConfig(cases []TuneCase, weights Weights, k int, lambda float64, useMMR bool) Metrics {
	var total Metrics
	for _, tc := range cases {
		results := rankTuneCandidates(tc.Candidates, weights, k, lambda, useMMR)
		metrics := Evaluate(tc.Query, results, k)
		total.RecallAtK += metrics.RecallAtK
		total.PrecisionAtK += metrics.PrecisionAtK
		total.MRR += metrics.MRR
		total.NDCGAtK += metrics.NDCGAtK
		total.UniqueTopicAtK += metrics.UniqueTopicAtK
	}
	count := float64(len(cases))
	return Metrics{
		RecallAtK:      total.RecallAtK / count,
		PrecisionAtK:   total.PrecisionAtK / count,
		MRR:            total.MRR / count,
		NDCGAtK:        total.NDCGAtK / count,
		UniqueTopicAtK: int(float64(total.UniqueTopicAtK) / count),
	}
}

func rankTuneCandidates(candidates []TuneCandidate, weights Weights, k int, lambda float64, useMMR bool) []RankedResult {
	items := make([]MMRItem, 0, len(candidates))
	for _, candidate := range candidates {
		items = append(items, MMRItem{
			ID:        candidate.ID,
			Score:     weightedSignals(candidate.Signals, weights),
			Embedding: candidate.Embedding,
			Topic:     candidate.Topic,
		})
	}
	slices.SortStableFunc(items, func(a, b MMRItem) int {
		if a.Score != b.Score {
			return cmp.Compare(b.Score, a.Score)
		}
		return cmp.Compare(a.ID, b.ID)
	})
	if useMMR {
		items = MMR(items, k, lambda)
	} else if len(items) > k {
		items = items[:k]
	}
	results := make([]RankedResult, 0, len(items))
	for _, item := range items {
		results = append(results, RankedResult{ID: item.ID, Score: item.Score, Topic: item.Topic})
	}
	return results
}

func weightedSignals(signals ScoreSignals, weights Weights) float64 {
	return weights.Embedding*signals.Embedding +
		weights.Keyword*signals.Keyword +
		weights.Filename*signals.Filename +
		weights.Recency*signals.Recency +
		weights.Importance*signals.Importance
}

func rejectTuneResult(metrics Metrics, constraints TuneConstraints) (bool, string) {
	switch {
	case metrics.RecallAtK < constraints.MinRecallAtK:
		return true, "recall_at_k"
	case metrics.NDCGAtK < constraints.MinNDCGAtK:
		return true, "ndcg_at_k"
	case metrics.UniqueTopicAtK < constraints.MinUniqueTopicsAtK:
		return true, "unique_topics_at_k"
	default:
		return false, ""
	}
}

func tuneResultBetter(a, b TuneResult) bool {
	if a.Metrics.NDCGAtK != b.Metrics.NDCGAtK {
		return a.Metrics.NDCGAtK > b.Metrics.NDCGAtK
	}
	if a.Metrics.RecallAtK != b.Metrics.RecallAtK {
		return a.Metrics.RecallAtK > b.Metrics.RecallAtK
	}
	if a.Metrics.MRR != b.Metrics.MRR {
		return a.Metrics.MRR > b.Metrics.MRR
	}
	if a.Metrics.PrecisionAtK != b.Metrics.PrecisionAtK {
		return a.Metrics.PrecisionAtK > b.Metrics.PrecisionAtK
	}
	if a.Metrics.UniqueTopicAtK != b.Metrics.UniqueTopicAtK {
		return a.Metrics.UniqueTopicAtK > b.Metrics.UniqueTopicAtK
	}
	if a.UsedMMR != b.UsedMMR {
		return !a.UsedMMR
	}
	if a.MMRLambda != b.MMRLambda {
		return a.MMRLambda > b.MMRLambda
	}
	return weightKeyLess(a.Weights, b.Weights)
}

func weightKeyLess(a, b Weights) bool {
	aValues := [5]float64{a.Embedding, a.Keyword, a.Filename, a.Recency, a.Importance}
	bValues := [5]float64{b.Embedding, b.Keyword, b.Filename, b.Recency, b.Importance}
	for i := range aValues {
		if aValues[i] != bValues[i] {
			return aValues[i] < bValues[i]
		}
	}
	return false
}

package retrieval

import (
	"cmp"
	"math"
	"slices"
)

// Weights configures relative importance of scoring signals.
//
// Recency and Importance are optional salience axes orthogonal to textual
// similarity. They default to 0 (DefaultWeights/AdaptiveWeights leave them
// unset), so callers that supply only Embedding/Keyword/Filename signals
// produce exactly the same CombinedScore as before these fields existed.
type Weights struct {
	Embedding  float64
	Keyword    float64
	Filename   float64
	Recency    float64
	Importance float64
}

// DefaultWeights provides sensible defaults for file organization.
func DefaultWeights() Weights {
	return Weights{Embedding: 0.63, Keyword: 0.27, Filename: 0.10}
}

// Result represents a scored item.
//
// RecencyScore and ImportanceScore are caller-supplied, already-normalized
// 0..1 values. Unlike EmbeddingScore/KeywordScore/FilenameScore they are NOT
// corpus min-max calibrated by Rerank: importance is an absolute salience tier
// and recency is an absolute time-decay, so corpus-relative rescaling would
// distort their meaning. Map your own importance tier (e.g. 1..5) or timestamp
// age (see RecencyFromAge) into 0..1 before assigning.
type Result struct {
	ID              string
	IndexIdentity   string
	DocumentID      string
	Content         string
	Filename        string
	Metadata        map[string]any
	Embedding       []float64
	EmbeddingScore  float64
	KeywordScore    float64
	FilenameScore   float64
	RecencyScore    float64
	ImportanceScore float64
	CombinedScore   float64
}

// ScoreSignals groups retrieval signal values.
type ScoreSignals struct {
	Embedding  float64
	Keyword    float64
	Filename   float64
	Recency    float64
	Importance float64
}

// SignalPresence reports which signals had usable inputs for a scored result.
type SignalPresence struct {
	Embedding  bool
	Keyword    bool
	Filename   bool
	Recency    bool
	Importance bool
}

// ScoreTrace explains how a result's CombinedScore was produced.
//
// Raw contains pre-calibration text/vector scores plus caller-supplied salience
// scores. Calibrated contains the values actually multiplied by Weights.
// Contributions is Calibrated multiplied by Weights per signal.
type ScoreTrace struct {
	ID              string
	QueryTokenCount int
	AdaptiveWeights bool
	Present         SignalPresence
	Weights         Weights
	Raw             ScoreSignals
	Calibrated      ScoreSignals
	Contributions   ScoreSignals
	CombinedScore   float64
}

// Scorer computes hybrid relevance scores.
type Scorer struct {
	weights         Weights
	adaptiveWeights bool
}

// NewScorer constructs a Scorer. When weights use the default text-signal mix,
// Rerank adapts Embedding/Keyword/Filename weights by query token length;
// explicit custom text weights are honored as-is.
func NewScorer(weights Weights) *Scorer {
	return &Scorer{weights: weights, adaptiveWeights: true}
}

// NewScorerWithOptions constructs a Scorer and lets callers disable adaptive
// weighting even when using DefaultWeights.
func NewScorerWithOptions(weights Weights, adaptiveWeights bool) *Scorer {
	return &Scorer{weights: weights, adaptiveWeights: adaptiveWeights}
}

// AdaptiveWeights computes weights by query token count.
func AdaptiveWeights(queryTokenCount int) Weights {
	filenameWeight := 0.10
	if queryTokenCount <= 2 {
		return Weights{Embedding: 0.45, Keyword: 0.45, Filename: filenameWeight}
	}
	if queryTokenCount >= 6 {
		return Weights{Embedding: 0.765, Keyword: 0.135, Filename: filenameWeight}
	}
	return DefaultWeights()
}

// RecencyFromAge maps an age to a 0..1 freshness score via exponential decay:
// 2^(-age/halfLife). An item exactly halfLife old scores 0.5; brand-new scores
// ~1.0; very old asymptotes toward 0. Both arguments are in the same time
// unit (e.g. seconds).
//
// Guards: age <= 0 returns 1.0 (treat future/now as fully fresh); halfLife <= 0
// returns 1.0 (no decay configured -> no penalty).
// Use the result as Result.ImportanceScore's sibling: assign to RecencyScore.
func RecencyFromAge(age, halfLife float64) float64 {
	if halfLife <= 0 || age <= 0 {
		return 1.0
	}
	return math.Exp2(-age / halfLife)
}

// Rerank scores and sorts results by combined score (descending).
func (s *Scorer) Rerank(queryEmbedding []float64, queryText string, results []*Result) []*Result {
	ranked, _ := s.rerank(queryEmbedding, queryText, results, false)
	return ranked
}

// RerankWithTrace scores and sorts results, returning one trace per ranked result
// in the same order as the returned result slice.
func (s *Scorer) RerankWithTrace(queryEmbedding []float64, queryText string, results []*Result) ([]*Result, []ScoreTrace) {
	return s.rerank(queryEmbedding, queryText, results, true)
}

func (s *Scorer) rerank(queryEmbedding []float64, queryText string, results []*Result, trace bool) ([]*Result, []ScoreTrace) {
	if len(results) == 0 {
		return results, nil
	}

	queryTokenCount := len(tokenize(queryText))
	weights, adaptiveWeights := s.effectiveWeights(queryTokenCount)

	channels := scoreChannels{}
	var tracesByResult map[*Result]scoreTraceState
	if trace {
		tracesByResult = make(map[*Result]scoreTraceState, len(results))
	}

	for _, result := range results {
		rawSignals, signalPresence := computeAndStoreRawSignals(queryEmbedding, queryText, result, &channels)
		if trace {
			tracesByResult[result] = scoreTraceState{raw: rawSignals, present: signalPresence}
		}
	}

	calibrateScores(results, channels)

	for _, result := range results {
		result.CombinedScore = weightedScore(weights, resultScoreSignals(result))
	}

	slices.SortFunc(results, func(a, b *Result) int {
		return cmp.Compare(b.CombinedScore, a.CombinedScore)
	})

	if !trace {
		return results, nil
	}

	traces := make([]ScoreTrace, 0, len(results))
	for _, result := range results {
		state := tracesByResult[result]
		calibrated := resultScoreSignals(result)
		traces = append(traces, ScoreTrace{
			ID:              result.ID,
			QueryTokenCount: queryTokenCount,
			AdaptiveWeights: adaptiveWeights,
			Present:         state.present,
			Weights:         weights,
			Raw:             state.raw,
			Calibrated:      calibrated,
			Contributions:   weightedSignalContributions(weights, calibrated),
			CombinedScore:   result.CombinedScore,
		})
	}

	return results, traces
}

// Score computes the combined score from raw (uncalibrated) component scores.
// The returned CombinedScore must NOT be compared against scores produced by
// Rerank, which applies corpus-relative min-max calibration before scoring.
func (s *Scorer) Score(queryEmbedding []float64, queryText string, result *Result) float64 {
	computeAndStoreRawSignals(queryEmbedding, queryText, result, nil)
	result.CombinedScore = weightedScore(s.weights, resultScoreSignals(result))
	return result.CombinedScore
}

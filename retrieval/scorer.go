package retrieval

import (
	"cmp"
	"container/heap"
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
// age (see RecencyFromAge) into 0..1 before assigning. Explain is ephemeral
// facade output and is nil unless reliquary.WithExplain was requested.
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
	Explain         *SearchExplanation
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
// Guards: NaN input returns 0.0; age <= 0 returns 1.0 (treat future/now as
// fully fresh); halfLife <= 0 returns 1.0 (no decay configured -> no penalty).
// Indeterminate exponential results such as +Inf/+Inf also return 0.0.
// Use the result as Result.ImportanceScore's sibling: assign to RecencyScore.
func RecencyFromAge(age, halfLife float64) float64 {
	if math.IsNaN(age) || math.IsNaN(halfLife) {
		return 0
	}
	if halfLife <= 0 || age <= 0 {
		return 1.0
	}
	return clamp01(math.Exp2(-age / halfLife))
}

// Rerank scores and sorts results by combined score (descending).
func (s *Scorer) Rerank(queryEmbedding []float64, queryText string, results []*Result) []*Result {
	ranked, _ := s.rerank(queryEmbedding, queryText, results, false, false)
	return ranked
}

// RerankWithTrace scores and sorts results, returning one trace per ranked result
// in the same order as the returned result slice.
func (s *Scorer) RerankWithTrace(queryEmbedding []float64, queryText string, results []*Result) ([]*Result, []ScoreTrace) {
	return s.rerank(queryEmbedding, queryText, results, true, false)
}

func (s *Scorer) rerank(queryEmbedding []float64, queryText string, results []*Result, trace, tieByID bool) ([]*Result, []ScoreTrace) {
	return s.scoreResults(queryEmbedding, queryText, results, trace, tieByID, true)
}

// RerankTopByID scores the entire pool before selecting a bounded score/ID
// prefix. A non-positive or non-limiting limit retains the full ranking.
func (s *Scorer) RerankTopByID(queryEmbedding []float64, queryText string, results []*Result, limit int) []*Result {
	if limit <= 0 || limit >= len(results) {
		ranked, _ := s.rerank(queryEmbedding, queryText, results, false, true)
		return ranked
	}
	// Corpus-relative calibration must include candidates outside the prefix.
	scored, _ := s.scoreResults(queryEmbedding, queryText, results, false, true, false)
	best := make(worstFirstResults, 0, limit)
	for _, result := range scored {
		if len(best) < limit {
			heap.Push(&best, result)
		} else if compareScoreID(result, best[0]) < 0 {
			best[0] = result
			heap.Fix(&best, 0)
		}
	}
	slices.SortFunc(best, compareScoreID)
	return best
}

// compareScoreID matches the full ranking's float (including NaN) and ID order.
func compareScoreID(a, b *Result) int {
	if order := cmp.Compare(b.CombinedScore, a.CombinedScore); order != 0 {
		return order
	}
	return cmp.Compare(a.ID, b.ID)
}

type worstFirstResults []*Result

func (h worstFirstResults) Len() int           { return len(h) }
func (h worstFirstResults) Less(i, j int) bool { return compareScoreID(h[i], h[j]) > 0 }
func (h worstFirstResults) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *worstFirstResults) Push(x any)        { *h = append(*h, x.(*Result)) }
func (h *worstFirstResults) Pop() any {
	last := len(*h) - 1
	item := (*h)[last]
	*h = (*h)[:last]
	return item
}

// ScoreBatch applies the same corpus calibration as Rerank without changing
// the order of results. Unlike Score, its scores are corpus-relative.
func (s *Scorer) ScoreBatch(queryEmbedding []float64, queryText string, results []*Result) []*Result {
	scored, _ := s.scoreResults(queryEmbedding, queryText, results, false, false, false)
	return scored
}

func (s *Scorer) scoreResults(queryEmbedding []float64, queryText string, results []*Result, trace, tieByID, sortResults bool) ([]*Result, []ScoreTrace) {
	if len(results) == 0 {
		return results, nil
	}

	queryTokens := tokenize(queryText)
	queryTokenCount := len(queryTokens)
	weights, adaptiveWeights := s.effectiveWeights(queryTokenCount)

	channels := scoreChannels{}
	presenceByResult := make(map[*Result]SignalPresence, len(results))
	var tracesByResult map[*Result]scoreTraceState
	if trace {
		tracesByResult = make(map[*Result]scoreTraceState, len(results))
	}

	for _, result := range results {
		rawSignals, signalPresence := computeAndStoreRawSignals(queryEmbedding, queryTokens, result, &channels)
		presenceByResult[result] = signalPresence
		if trace {
			tracesByResult[result] = scoreTraceState{raw: rawSignals, present: signalPresence}
		}
	}

	calibrateScores(results, channels, presenceByResult)

	for _, result := range results {
		result.CombinedScore = weightedScore(weights, resultScoreSignals(result))
	}

	if sortResults {
		if tieByID {
			slices.SortFunc(results, compareScoreID)
		} else {
			slices.SortFunc(results, func(a, b *Result) int {
				return cmp.Compare(b.CombinedScore, a.CombinedScore)
			})
		}
	}

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
	computeAndStoreRawSignals(queryEmbedding, tokenize(queryText), result, nil)
	result.CombinedScore = weightedScore(s.weights, resultScoreSignals(result))
	return result.CombinedScore
}

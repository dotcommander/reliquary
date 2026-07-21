package retrieval

import (
	"math"

	"github.com/dotcommander/reliquary/vector"
)

// scoreChannels tracks which signal families were computed for this rerank call.
type scoreChannels struct {
	embedding bool
	keyword   bool
	filename  bool
}

type scoreTraceState struct {
	raw     ScoreSignals
	present SignalPresence
}

func (s *Scorer) effectiveWeights(queryTokenCount int) (Weights, bool) {
	weights := s.weights
	adaptiveWeights := false
	if s.adaptiveWeights && usesDefaultTextWeights(s.weights) {
		weights = AdaptiveWeights(queryTokenCount)
		weights.Recency = s.weights.Recency
		weights.Importance = s.weights.Importance
		adaptiveWeights = true
	}
	return weights, adaptiveWeights
}

func usesDefaultTextWeights(weights Weights) bool {
	defaults := DefaultWeights()
	return weights.Embedding == defaults.Embedding &&
		weights.Keyword == defaults.Keyword &&
		weights.Filename == defaults.Filename
}

func computeAndStoreRawSignals(queryEmbedding []float64, queryText string, result *Result, channels *scoreChannels) (ScoreSignals, SignalPresence) {
	var raw ScoreSignals
	var present SignalPresence
	result.EmbeddingScore = 0
	result.KeywordScore = 0
	result.FilenameScore = 0
	result.CombinedScore = 0
	hasText := queryText != ""
	hasEmbedding := len(queryEmbedding) > 0

	if hasEmbedding && len(result.Embedding) > 0 {
		result.EmbeddingScore = vectors.Cosine64(queryEmbedding, result.Embedding)
		raw.Embedding = result.EmbeddingScore
		present.Embedding = true
		if channels != nil {
			channels.embedding = true
		}
	}
	if hasText && result.Content != "" {
		result.KeywordScore = keywordOverlap(queryText, result.Content)
		raw.Keyword = result.KeywordScore
		present.Keyword = true
		if channels != nil {
			channels.keyword = true
		}
	}
	if hasText && result.Filename != "" {
		result.FilenameScore = FilenameOverlap(result.Filename, queryText)
		raw.Filename = result.FilenameScore
		present.Filename = true
		if channels != nil {
			channels.filename = true
		}
	}

	raw.Recency = result.RecencyScore
	raw.Importance = result.ImportanceScore
	present.Recency = result.RecencyScore != 0
	present.Importance = result.ImportanceScore != 0

	return raw, present
}

func calibrateScores(results []*Result, channels scoreChannels, presenceByResult map[*Result]SignalPresence) {
	if len(results) < 2 {
		return
	}

	var minEmb, maxEmb float64 = math.MaxFloat64, -math.MaxFloat64
	var minKwd, maxKwd float64 = math.MaxFloat64, -math.MaxFloat64
	var minFn, maxFn float64 = math.MaxFloat64, -math.MaxFloat64

	for _, result := range results {
		presence := presenceByResult[result]
		if channels.embedding && presence.Embedding {
			minEmb = min(minEmb, result.EmbeddingScore)
			maxEmb = max(maxEmb, result.EmbeddingScore)
		}
		if channels.keyword && presence.Keyword {
			minKwd = min(minKwd, result.KeywordScore)
			maxKwd = max(maxKwd, result.KeywordScore)
		}
		if channels.filename && presence.Filename {
			minFn = min(minFn, result.FilenameScore)
			maxFn = max(maxFn, result.FilenameScore)
		}
	}

	for _, result := range results {
		presence := presenceByResult[result]
		if channels.embedding && presence.Embedding {
			result.EmbeddingScore = normalizeValue(result.EmbeddingScore, minEmb, maxEmb)
		}
		if channels.keyword && presence.Keyword {
			result.KeywordScore = normalizeValue(result.KeywordScore, minKwd, maxKwd)
		}
		if channels.filename && presence.Filename {
			result.FilenameScore = normalizeValue(result.FilenameScore, minFn, maxFn)
		}
	}
}

func resultScoreSignals(result *Result) ScoreSignals {
	return ScoreSignals{
		Embedding:  result.EmbeddingScore,
		Keyword:    result.KeywordScore,
		Filename:   result.FilenameScore,
		Recency:    result.RecencyScore,
		Importance: result.ImportanceScore,
	}
}

func weightedSignalContributions(weights Weights, signals ScoreSignals) ScoreSignals {
	return ScoreSignals{
		Embedding:  weights.Embedding * signals.Embedding,
		Keyword:    weights.Keyword * signals.Keyword,
		Filename:   weights.Filename * signals.Filename,
		Recency:    weights.Recency * signals.Recency,
		Importance: weights.Importance * signals.Importance,
	}
}

func weightedScore(weights Weights, signals ScoreSignals) float64 {
	weighted := weightedSignalContributions(weights, signals)
	return weighted.Embedding + weighted.Keyword + weighted.Filename + weighted.Recency + weighted.Importance
}

func normalizeValue(val, minValue, maxValue float64) float64 {
	if maxValue-minValue < 1e-10 {
		return 0.5
	}
	return (val - minValue) / (maxValue - minValue)
}

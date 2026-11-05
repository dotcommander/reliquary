package retrieval

import (
	"cmp"
	"container/heap"
	"slices"

	vectors "github.com/dotcommander/reliquary/vector"
)

// RawCandidate retains the uncalibrated signals captured during index selection.
type RawCandidate struct {
	Result  *Result
	Raw     ScoreSignals
	Present SignalPresence
}

// SelectWithRaw scores the full pool and selects the same score/ID prefix as
// RerankTopByID, retaining the original signals before calibration.
func (s *Scorer) SelectWithRaw(queryEmbedding []float64, queryText string, results []*Result, limit int) []RawCandidate {
	if len(results) == 0 {
		return nil
	}
	tokens := tokenize(queryText)
	weights, _ := s.effectiveWeights(len(tokens))
	prepared := vectors.Prepare64(queryEmbedding)
	channels := scoreChannels{}
	presence := make(map[*Result]SignalPresence, len(results))
	raw := make(map[*Result]ScoreSignals, len(results))
	for _, result := range results {
		raw[result], presence[result] = computeAndStoreRawSignals(queryEmbedding, prepared, tokens, result, &channels)
	}
	calibrateScores(results, channels, presence)
	for _, result := range results {
		result.CombinedScore = weightedScore(weights, resultScoreSignals(result))
	}
	if limit > 0 && limit < len(results) {
		best := make(worstFirstResults, 0, limit)
		for _, result := range results {
			if len(best) < limit {
				heap.Push(&best, result)
			} else if compareScoreID(result, best[0]) < 0 {
				best[0] = result
				heap.Fix(&best, 0)
			}
		}
		results = best
	}
	slices.SortFunc(results, func(a, b *Result) int {
		if n := cmp.Compare(b.CombinedScore, a.CombinedScore); n != 0 {
			return n
		}
		return cmp.Compare(a.ID, b.ID)
	})
	selected := make([]RawCandidate, len(results))
	for n, result := range results {
		selected[n] = RawCandidate{Result: result, Raw: raw[result], Present: presence[result]}
	}
	return selected
}

// CompleteLaneSignals fills only the signals absent from a vector-only or
// text-only selection query, before returning the selected candidate.
func CompleteLaneSignals(candidate *RawCandidate, vector []float64, text string, vectorLane bool) {
	if vectorLane {
		tokens := tokenize(text)
		if len(tokens) > 0 {
			content := tokenize(candidate.Result.Content)
			filename := tokenize(candidate.Result.Filename)
			if len(content) > 0 {
				candidate.Raw.Keyword = overlapRatioScan(tokens, content)
				candidate.Present.Keyword = true
			}
			if len(filename) > 0 {
				candidate.Raw.Filename = overlapRatioScan(filename, tokens)
				candidate.Present.Filename = true
			}
		}
	} else if len(vector) > 0 && len(candidate.Result.Embedding) > 0 {
		candidate.Raw.Embedding = vectors.Cosine64Prepared(vector, vectors.Prepare64(vector), candidate.Result.Embedding, vectors.Prepare64(candidate.Result.Embedding))
		candidate.Present.Embedding = true
	}
}

// RecalibrateRaw applies facade calibration to a retained subset without
// repeating tokenization or vector similarity. It preserves stable score ties.
func (s *Scorer) RecalibrateRaw(queryText string, candidates []RawCandidate, trace, sortResults bool) ([]*Result, []ScoreTrace) {
	if len(candidates) == 0 {
		return nil, nil
	}
	count := len(tokenize(queryText))
	weights, adaptive := s.effectiveWeights(count)
	results := make([]*Result, len(candidates))
	presence := make(map[*Result]SignalPresence, len(candidates))
	raw := make(map[*Result]ScoreSignals, len(candidates))
	channels := scoreChannels{}
	for n, candidate := range candidates {
		result := candidate.Result
		results[n] = result
		raw[result] = candidate.Raw
		presence[result] = candidate.Present
		result.EmbeddingScore = candidate.Raw.Embedding
		result.KeywordScore = candidate.Raw.Keyword
		result.FilenameScore = candidate.Raw.Filename
		channels.embedding = channels.embedding || candidate.Present.Embedding
		channels.keyword = channels.keyword || candidate.Present.Keyword
		channels.filename = channels.filename || candidate.Present.Filename
	}
	calibrateScores(results, channels, presence)
	tracesByResult := make(map[*Result]ScoreTrace, len(results))
	for _, result := range results {
		calibrated := resultScoreSignals(result)
		result.CombinedScore = weightedScore(weights, calibrated)
		if trace {
			tracesByResult[result] = ScoreTrace{ID: result.ID, QueryTokenCount: count, AdaptiveWeights: adaptive, Present: presence[result], Weights: weights, Raw: raw[result], Calibrated: calibrated, Contributions: weightedSignalContributions(weights, calibrated), CombinedScore: result.CombinedScore}
		}
	}
	if sortResults {
		slices.SortFunc(results, func(a, b *Result) int { return cmp.Compare(b.CombinedScore, a.CombinedScore) })
	}
	if !trace {
		return results, nil
	}
	traces := make([]ScoreTrace, len(results))
	for n, result := range results {
		traces[n] = tracesByResult[result]
	}
	return results, traces
}

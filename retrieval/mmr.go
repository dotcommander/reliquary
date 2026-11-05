package retrieval

import (
	"math"

	"github.com/dotcommander/reliquary/vector"
)

// MMRItem represents an item participating in MMR diversification.
type MMRItem struct {
	ID        string
	Score     float64
	Embedding []float64
	Topic     string
}

// MMRItems adapts scored retrieval results into MMR input items.
func MMRItems(results []*Result) []MMRItem {
	items := make([]MMRItem, 0, len(results))
	for _, r := range results {
		if r == nil {
			continue
		}
		items = append(items, MMRItem{
			ID:        r.ID,
			Score:     r.CombinedScore,
			Embedding: r.Embedding,
			Topic:     r.Filename,
		})
	}
	return items
}

// Diversify applies MMR to ranked retrieval results and returns the matching
// result pointers in diversified order.
func Diversify(results []*Result, k int, lambda float64) []*Result {
	out, _ := diversify(results, k, lambda, false)
	return out
}

// DiversifyWithTrace applies the same MMR selection as Diversify and returns
// one rank-aligned explanation for each selected result.
func DiversifyWithTrace(results []*Result, k int, lambda float64) ([]*Result, []MMRExplanation) {
	return diversify(results, k, lambda, true)
}

func diversify(results []*Result, k int, lambda float64, trace bool) ([]*Result, []MMRExplanation) {
	candidates := make([]*Result, 0, len(results))
	for _, result := range results {
		if result != nil {
			candidates = append(candidates, result)
		}
	}
	return selectMMR(candidates, k, lambda, func(result *Result) float64 {
		return result.CombinedScore
	}, func(result *Result) []float64 {
		return result.Embedding
	}, trace)
}

func MMR(items []MMRItem, k int, lambda float64) []MMRItem {
	selected, _ := selectMMR(items, k, lambda, func(item MMRItem) float64 {
		return item.Score
	}, func(item MMRItem) []float64 {
		return item.Embedding
	}, false)
	return selected
}

// mmrCandidate pairs an item with its embedding and precomputed norm so
// pairwise similarity reuses prepared norms instead of recomputing them.
type mmrCandidate[T any] struct {
	item   T
	vector []float64
	norm   vectors.Prepared64
}

func selectMMR[T any](items []T, k int, lambda float64, relevance func(T) float64, embeddingOf func(T) []float64, trace bool) ([]T, []MMRExplanation) {
	if k <= 0 || len(items) == 0 {
		return nil, nil
	}
	lambda = clamp01(lambda)
	remaining := make([]mmrCandidate[T], 0, len(items))
	for _, item := range items {
		vector := embeddingOf(item)
		remaining = append(remaining, mmrCandidate[T]{item: item, vector: vector, norm: vectors.Prepare64(vector)})
	}
	maxSimilarities := make([]float64, len(remaining))
	// selectedCandidates is the only selection state; the returned items are
	// its projection, so the two can never drift apart.
	selectedCandidates := make([]mmrCandidate[T], 0, min(k, len(items)))
	var traces []MMRExplanation
	if trace {
		traces = make([]MMRExplanation, 0, min(k, len(items)))
	}
	for len(remaining) > 0 && len(selectedCandidates) < k {
		// Only the newest selection can raise a remaining item's maximum.
		if len(selectedCandidates) > 0 && (lambda != 1 || trace) {
			// At lambda=1 an untraced selection uses only relevance; traced
			// selections still need the exact similarity and penalty values.
			newest := selectedCandidates[len(selectedCandidates)-1]
			for i, item := range remaining {
				if sim := vectors.Cosine64Prepared(item.vector, item.norm, newest.vector, newest.norm); sim > maxSimilarities[i] {
					maxSimilarities[i] = sim
				}
			}
		}
		bestIdx := 0
		bestScore := math.Inf(-1)
		bestSimilarity := 0.0
		hasBest := false
		for i, item := range remaining {
			itemSimilarity := maxSimilarities[i]
			score := lambda*relevance(item.item) - (1-lambda)*itemSimilarity
			if i == 0 {
				bestSimilarity = itemSimilarity
			}
			if math.IsNaN(score) {
				continue
			}
			if !hasBest || score > bestScore {
				hasBest = true
				bestScore = score
				bestIdx = i
				bestSimilarity = itemSimilarity
			}
		}
		chosen := remaining[bestIdx]
		selectedCandidates = append(selectedCandidates, chosen)
		if trace {
			relevanceScore := relevance(chosen.item)
			relevanceContribution := lambda * relevanceScore
			penalty := -(1 - lambda) * bestSimilarity
			traces = append(traces, MMRExplanation{
				Lambda:                lambda,
				Relevance:             relevanceScore,
				MaxSimilarity:         bestSimilarity,
				RelevanceContribution: relevanceContribution,
				Penalty:               penalty,
				SelectionScore:        relevanceContribution + penalty,
			})
		}
		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
		maxSimilarities = append(maxSimilarities[:bestIdx], maxSimilarities[bestIdx+1:]...)
	}
	selected := make([]T, len(selectedCandidates))
	for n, candidate := range selectedCandidates {
		selected[n] = candidate.item
	}
	return selected, traces
}

func maxSimilarity(item MMRItem, selected []MMRItem) float64 {
	maxSim := 0.0
	if len(item.Embedding) == 0 {
		return 0
	}
	for _, existing := range selected {
		if len(existing.Embedding) == 0 {
			continue
		}
		sim := vectors.Cosine64(item.Embedding, existing.Embedding)
		if sim > maxSim {
			maxSim = sim
		}
	}
	return maxSim
}

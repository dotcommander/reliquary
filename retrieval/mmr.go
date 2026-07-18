package retrieval

import "github.com/dotcommander/reliquary/vector"

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
	items := MMR(MMRItems(results), k, lambda)
	byID := make(map[string]*Result, len(results))
	for _, r := range results {
		if r != nil {
			byID[r.ID] = r
		}
	}
	out := make([]*Result, 0, len(items))
	for _, item := range items {
		if r := byID[item.ID]; r != nil {
			out = append(out, r)
		}
	}
	return out
}

func MMR(items []MMRItem, k int, lambda float64) []MMRItem {
	if k <= 0 || len(items) == 0 {
		return nil
	}
	lambda = clamp01(lambda)
	remaining := append([]MMRItem(nil), items...)
	selected := make([]MMRItem, 0, min(k, len(items)))
	for len(remaining) > 0 && len(selected) < k {
		bestIdx := 0
		bestScore := -2.0
		for i, item := range remaining {
			score := lambda*item.Score - (1-lambda)*maxSimilarity(item, selected)
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}
		selected = append(selected, remaining[bestIdx])
		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
	}
	return selected
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

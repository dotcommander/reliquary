package vectors

import (
	"container/heap"
	"slices"
)

// TopKMaxIndices returns the indices of the k largest values in scores, ordered
// largest-first. k is clamped to len(scores); k<=0 returns an empty slice. Equal
// scores are ordered by ascending index (stable). Runs in O(n log k) time, O(k) space.
func TopKMaxIndices(scores []float32, k int) []int { return topK(scores, k, true) }

// TopKMinIndices returns the indices of the k smallest values in scores, ordered
// smallest-first. Same clamping and stability rules as TopKMaxIndices.
func TopKMinIndices(scores []float32, k int) []int { return topK(scores, k, false) }

func topK(scores []float32, k int, max bool) []int {
	if k <= 0 || len(scores) == 0 {
		return []int{}
	}
	if k > len(scores) {
		k = len(scores)
	}

	h := &idxHeap{max: max}
	for i, score := range scores {
		if h.Len() < k {
			heap.Push(h, idxItem{index: i, score: score})
			continue
		}

		if shouldReplace(score, i, h.Peek(), max) {
			heap.Pop(h)
			heap.Push(h, idxItem{index: i, score: score})
		}
	}

	out := make([]int, 0, k)
	for h.Len() > 0 {
		out = append(out, heap.Pop(h).(idxItem).index)
	}
	slices.Reverse(out)
	return out
}

func shouldReplace(score float32, index int, existing idxItem, max bool) bool {
	if score != existing.score {
		if max {
			return score > existing.score
		}
		return score < existing.score
	}
	return index < existing.index
}

type idxItem struct {
	score float32
	index int
}

type idxHeap struct {
	items []idxItem
	max   bool
}

func (h *idxHeap) Len() int           { return len(h.items) }
func (h *idxHeap) Less(i, j int) bool { return h.less(h.items[i], h.items[j]) }
func (h *idxHeap) Swap(i, j int)      { h.items[i], h.items[j] = h.items[j], h.items[i] }

func (h *idxHeap) Push(x any) {
	h.items = append(h.items, x.(idxItem))
}

func (h *idxHeap) Pop() any {
	old := h.items
	n := len(old)
	v := old[n-1]
	h.items = old[:n-1]
	return v
}

func (h *idxHeap) Peek() idxItem {
	return h.items[0]
}

func (h *idxHeap) less(a, b idxItem) bool {
	if a.score == b.score {
		return a.index > b.index
	}

	if h.max {
		// min-heap collects the k largest: root is the eviction threshold.
		return a.score < b.score
	}
	// max-heap collects the k smallest: root is the eviction threshold.
	return a.score > b.score
}

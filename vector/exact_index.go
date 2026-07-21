package vectors

import (
	"container/heap"
	"slices"
	"sync"
)

// IndexChunk defines the location of a chunk's vector in the index arena.
type IndexChunk struct {
	Group      string // e.g. EntryID
	ChunkIndex int    // 0-based index of this chunk within the Group
	Offset     int    // Byte offset into the contiguous arena slice
	Length     int    // Byte length of this vector in the arena
}

// SearchResult holds a query match.
type SearchResult struct {
	Group      string
	ChunkIndex int
	Score      float64
}

// IndexKey identifies one indexed chunk without tying vectors to caller storage.
type IndexKey struct {
	Group      string
	ChunkIndex int
}

// IndexBuildReport describes rows accepted or skipped while building an index.
type IndexBuildReport struct {
	InputRows              int
	IndexedRows            int
	SkippedBadSpan         int
	SkippedBadBlob         int
	SkippedMissingMetadata int
	SkippedDuplicateKey    int
	DimensionMismatch      int
	MedianError            string
	QuantizeError          string
}

type scoredItem struct {
	group      string
	chunkIndex int
	score      float64
}

type minScoredHeap []scoredItem

func (h minScoredHeap) Len() int           { return len(h) }
func (h minScoredHeap) Less(i, j int) bool { return scoredItemBetter(h[j], h[i]) }
func (h minScoredHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minScoredHeap) Push(x any)        { *h = append(*h, x.(scoredItem)) }
func (h *minScoredHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// scoredItemBetter defines the result order used both while retaining the
// top-K cutoff and when returning results.
func scoredItemBetter(a, b scoredItem) bool {
	if a.score != b.score {
		return a.score > b.score
	}
	if a.group != b.group {
		return a.group < b.group
	}
	return a.chunkIndex < b.chunkIndex
}

// ExactIndex is a thread-safe, in-memory flat vector index.
// It stores L2-normalized float32 vectors packed in a single contiguous byte arena
// to minimize GC allocation overhead.
type ExactIndex struct {
	mu                sync.RWMutex
	dims              int
	chunks            []IndexChunk
	arena             []byte
	groupChunkIndexes map[string][]int
	chunkKeyIndexes   map[IndexKey]int
}

// NewExactIndex constructs an ExactIndex. It snapshots arena, so callers may
// mutate or release their slice after this function returns.
func NewExactIndex(dims int, chunks []IndexChunk, arena []byte) *ExactIndex {
	idx, _ := NewExactIndexChecked(dims, chunks, arena)
	return idx
}

// NewExactIndexChecked constructs an ExactIndex and reports skipped or
// suspicious rows. Rows containing non-finite vector values are skipped and
// counted in SkippedBadBlob. It snapshots arena, so callers may mutate or
// release their slice after this function returns.
func NewExactIndexChecked(dims int, chunks []IndexChunk, arena []byte) (*ExactIndex, IndexBuildReport) {
	report := IndexBuildReport{InputRows: len(chunks)}

	// Drop chunks whose arena span is out of bounds so Search paths cannot panic.
	valid := make([]IndexChunk, 0, len(chunks))
	seenKeys := make(map[IndexKey]struct{}, len(chunks))
	for _, c := range chunks {
		if !validArenaSpan(c, len(arena)) {
			report.SkippedBadSpan++
			continue
		}
		if dims <= 0 || c.Length != dims*4 {
			report.DimensionMismatch++
			continue
		}
		if !finiteFloat32Blob(arena[c.Offset : c.Offset+c.Length]) {
			report.SkippedBadBlob++
			continue
		}
		key := IndexKey{Group: c.Group, ChunkIndex: c.ChunkIndex}
		if _, exists := seenKeys[key]; exists {
			report.SkippedDuplicateKey++
			continue
		}
		seenKeys[key] = struct{}{}
		valid = append(valid, c)
	}

	idx := &ExactIndex{
		dims:              dims,
		chunks:            valid,
		arena:             slices.Clone(arena),
		groupChunkIndexes: make(map[string][]int),
		chunkKeyIndexes:   make(map[IndexKey]int),
	}

	for i, c := range valid {
		idx.groupChunkIndexes[c.Group] = append(idx.groupChunkIndexes[c.Group], i)
		idx.chunkKeyIndexes[IndexKey{Group: c.Group, ChunkIndex: c.ChunkIndex}] = i
	}
	report.IndexedRows = len(valid)
	return idx, report
}

func validArenaSpan(c IndexChunk, arenaLen int) bool {
	if c.Offset < 0 || c.Length < 0 || c.Length > arenaLen {
		return false
	}
	return c.Offset <= arenaLen-c.Length
}

// Len returns the number of chunks stored.
func (idx *ExactIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.chunks)
}

// Clear clears all internal slices and maps.
func (idx *ExactIndex) Clear() {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.chunks = nil
	idx.arena = nil
	idx.groupChunkIndexes = nil
	idx.chunkKeyIndexes = nil
}

// prepareSearch applies the default result limit and reports whether the
// index holds any chunks to search. Call while holding idx.mu (R or L).
func (idx *ExactIndex) prepareSearch(limit int) (int, bool) {
	if len(idx.chunks) == 0 {
		return 0, false
	}
	if limit <= 0 {
		limit = 10
		return limit, true
	}
	return limit, true
}

// Search scores all vectors against the query and returns the top-K matches.
// Returns false if the index is empty or the query is invalid.
func (idx *ExactIndex) Search(queryVec []float32, limit int, minSimilarity float64) ([]SearchResult, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(queryVec) != idx.dims || !finiteFloat32s(queryVec) {
		return nil, false
	}
	limit, ok := idx.prepareSearch(limit)
	if !ok {
		return nil, false
	}

	h := &minScoredHeap{}
	for _, c := range idx.chunks {
		idx.scoreChunk(queryVec, limit, minSimilarity, h, c)
	}
	return heapToResults(h), true
}

// SearchFiltered scores only vectors belonging to the specified groups.
// Returns false if the index is empty or the query is invalid.
func (idx *ExactIndex) SearchFiltered(
	queryVec []float32,
	limit int,
	minSimilarity float64,
	groups []string,
) ([]SearchResult, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(queryVec) != idx.dims || !finiteFloat32s(queryVec) {
		return nil, false
	}
	limit, ok := idx.prepareSearch(limit)
	if !ok {
		return nil, false
	}

	indexes := make([]int, 0, len(groups))
	seenGroups := make(map[string]struct{}, len(groups))
	for _, g := range groups {
		if _, seen := seenGroups[g]; seen {
			continue
		}
		seenGroups[g] = struct{}{}
		if groupIndexes, ok := idx.groupChunkIndexes[g]; ok {
			indexes = append(indexes, groupIndexes...)
		}
	}
	if len(indexes) == 0 {
		return nil, true
	}
	slices.Sort(indexes)

	h := &minScoredHeap{}
	for _, chunkIndex := range indexes {
		idx.scoreChunk(queryVec, limit, minSimilarity, h, idx.chunks[chunkIndex])
	}
	return heapToResults(h), true
}

// SearchKeys scores only the chunks identified by keys. Duplicate and missing
// keys are ignored. Returns false if the index is empty or the query is
// invalid.
func (idx *ExactIndex) SearchKeys(
	queryVec []float32,
	limit int,
	minSimilarity float64,
	keys []IndexKey,
) ([]SearchResult, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(queryVec) != idx.dims || !finiteFloat32s(queryVec) {
		return nil, false
	}
	limit, ok := idx.prepareSearch(limit)
	if !ok {
		return nil, false
	}

	indexes := make([]int, 0, len(keys))
	seen := make(map[IndexKey]struct{}, len(keys))
	for _, key := range keys {
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		if chunkIndex, ok := idx.chunkKeyIndexes[key]; ok {
			indexes = append(indexes, chunkIndex)
		}
	}
	if len(indexes) == 0 {
		return nil, true
	}
	slices.Sort(indexes)

	h := &minScoredHeap{}
	for _, chunkIndex := range indexes {
		idx.scoreChunk(queryVec, limit, minSimilarity, h, idx.chunks[chunkIndex])
	}
	return heapToResults(h), true
}

// SearchGroupsByMaxPool pools the best similarity score per group across all
// chunks and returns the top-K best matching groups. Returns false if the index
// is empty or the query is invalid.
func (idx *ExactIndex) SearchGroupsByMaxPool(queryVec []float32, limit int) ([]SearchResult, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(queryVec) != idx.dims || !finiteFloat32s(queryVec) {
		return nil, false
	}
	limit, ok := idx.prepareSearch(limit)
	if !ok {
		return nil, false
	}

	type groupBest struct {
		score float64
	}
	best := make(map[string]groupBest, len(idx.groupChunkIndexes))
	for _, c := range idx.chunks {
		score := float64(DotFromBlob(queryVec, idx.arena[c.Offset:c.Offset+c.Length]))
		if b, ok := best[c.Group]; !ok || score > b.score {
			best[c.Group] = groupBest{score: score}
		}
	}
	if len(best) == 0 {
		return nil, true
	}

	h := &minScoredHeap{}
	for group, b := range best {
		item := scoredItem{group: group, score: b.score, chunkIndex: -1}
		if h.Len() < limit {
			heap.Push(h, item)
		} else if scoredItemBetter(item, (*h)[0]) {
			(*h)[0] = item
			heap.Fix(h, 0)
		}
	}

	return heapToResults(h), true
}

func (idx *ExactIndex) scoreChunk(
	queryVec []float32,
	limit int,
	minSimilarity float64,
	h *minScoredHeap,
	c IndexChunk,
) {
	score := float64(DotFromBlob(queryVec, idx.arena[c.Offset:c.Offset+c.Length]))
	if minSimilarity > 0 && score < minSimilarity {
		return
	}

	item := scoredItem{group: c.Group, chunkIndex: c.ChunkIndex, score: score}
	if h.Len() < limit {
		heap.Push(h, item)
	} else if scoredItemBetter(item, (*h)[0]) {
		(*h)[0] = item
		heap.Fix(h, 0)
	}
}

func heapToResults(h *minScoredHeap) []SearchResult {
	top := make([]scoredItem, h.Len())
	for i := range top {
		top[i] = heap.Pop(h).(scoredItem)
	}
	if len(top) == 0 {
		return nil
	}
	slices.SortFunc(top, func(a, b scoredItem) int {
		if scoredItemBetter(a, b) {
			return -1
		}
		if scoredItemBetter(b, a) {
			return 1
		}
		return 0
	})

	results := make([]SearchResult, len(top))
	for i, c := range top {
		results[i] = SearchResult{
			Group:      c.group,
			ChunkIndex: c.chunkIndex,
			Score:      c.score,
		}
	}
	return results
}

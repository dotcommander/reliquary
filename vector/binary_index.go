package vectors

import (
	"container/heap"
	"slices"
	"sync"
)

const binaryPreFilterK = 100

// BinaryIndexEntry maps a binary vector back to its source ID and chunk index.
type BinaryIndexEntry struct {
	Group      string
	ChunkIndex int
}

// HammingCandidate holds a candidate from the Hamming pre-filter stage.
type HammingCandidate struct {
	Group      string
	ChunkIndex int
	Hamming    int // Lower = more similar
}

type maxHammingHeap []HammingCandidate

func (h maxHammingHeap) Len() int           { return len(h) }
func (h maxHammingHeap) Less(i, j int) bool { return hammingCandidateWorse(h[i], h[j]) }
func (h maxHammingHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHammingHeap) Push(x any)        { *h = append(*h, x.(HammingCandidate)) }
func (h *maxHammingHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// BinaryIndex is a thread-safe, in-memory pre-filter for semantic search.
// It stores packed bit vectors for all chunks and provides fast Hamming-distance pre-filtering.
type BinaryIndex struct {
	mu      sync.RWMutex
	entries []BinaryIndexEntry
	vectors []BinaryVector
	medians []float32
	dims    int
}

// NewBinaryIndex builds a BinaryIndex from raw little-endian float32 blobs.
// Rows with invalid or non-finite blobs, missing group/chunk metadata, or
// duplicate keys are skipped. For duplicate keys, the first valid row wins.
func NewBinaryIndex(blobs [][]byte, groups []string, chunkIndices []int, dims int) *BinaryIndex {
	idx, _ := NewBinaryIndexChecked(blobs, groups, chunkIndices, dims)
	return idx
}

// NewBinaryIndexChecked builds a BinaryIndex and reports skipped rows and build
// errors. Rows are validated before their keys are reserved, so the first valid
// row for each (Group, ChunkIndex) key is retained.
func NewBinaryIndexChecked(blobs [][]byte, groups []string, chunkIndices []int, dims int) (*BinaryIndex, IndexBuildReport) {
	idx := &BinaryIndex{dims: dims}
	report := IndexBuildReport{InputRows: len(blobs)}

	if len(blobs) == 0 {
		return idx, report
	}

	vecs := make([][]float32, 0, len(blobs))
	entries := make([]BinaryIndexEntry, 0, len(blobs))
	seenKeys := make(map[IndexKey]struct{}, len(blobs))

	for i, blob := range blobs {
		if i >= len(groups) || i >= len(chunkIndices) {
			report.SkippedMissingMetadata++
			continue
		}
		vec := DecodeFloat32Vec(blob)
		if vec == nil {
			report.SkippedBadBlob++
			continue
		}
		if dims > 0 && len(vec) != dims {
			report.DimensionMismatch++
			continue
		}
		if !finiteFloat32s(vec) {
			report.SkippedBadBlob++
			continue
		}
		key := IndexKey{Group: groups[i], ChunkIndex: chunkIndices[i]}
		if _, exists := seenKeys[key]; exists {
			report.SkippedDuplicateKey++
			continue
		}
		seenKeys[key] = struct{}{}
		vecs = append(vecs, vec)
		entries = append(entries, BinaryIndexEntry{
			Group:      groups[i],
			ChunkIndex: chunkIndices[i],
		})
	}

	if len(vecs) == 0 {
		return idx, report
	}

	medians, err := ComputeMediansChecked(vecs)
	if err != nil {
		report.MedianError = err.Error()
		return idx, report
	}

	binVecs := make([]BinaryVector, len(vecs))
	for i, vec := range vecs {
		bv, err := Quantize(vec, medians)
		if err != nil {
			report.QuantizeError = err.Error()
			return idx, report
		}
		binVecs[i] = bv
	}

	idx.entries = entries
	idx.vectors = binVecs
	idx.medians = medians
	report.IndexedRows = len(entries)
	return idx, report
}

// SearchCandidates returns the best Hamming-distance candidates for exact re-ranking.
func (idx *BinaryIndex) SearchCandidates(queryVec []float32) ([]HammingCandidate, bool) {
	return idx.SearchCandidatesLimit(queryVec, binaryPreFilterK)
}

// SearchCandidatesLimit returns up to limit best Hamming-distance candidates
// for exact re-ranking. A limit larger than the index size is clamped; a
// non-positive limit returns an empty candidate slice for a valid query.
// Non-finite queries are rejected.
func (idx *BinaryIndex) SearchCandidatesLimit(queryVec []float32, limit int) ([]HammingCandidate, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.entries) == 0 {
		return nil, false
	}
	if !finiteFloat32s(queryVec) {
		return nil, false
	}

	queryBin, err := Quantize(queryVec, idx.medians)
	if err != nil {
		return nil, false
	}

	preFilterK := limit
	if preFilterK <= 0 {
		return []HammingCandidate{}, true
	}
	if preFilterK > len(idx.vectors) {
		preFilterK = len(idx.vectors)
	}

	h := &maxHammingHeap{}
	for i, v := range idx.vectors {
		candidate := HammingCandidate{
			Group:      idx.entries[i].Group,
			ChunkIndex: idx.entries[i].ChunkIndex,
			Hamming:    HammingDistance(queryBin, v),
		}
		if h.Len() < preFilterK {
			heap.Push(h, candidate)
		} else if hammingCandidateBetter(candidate, (*h)[0]) {
			(*h)[0] = candidate
			heap.Fix(h, 0)
		}
	}

	candidates := make([]HammingCandidate, h.Len())
	for i := len(candidates) - 1; i >= 0; i-- {
		candidates[i] = heap.Pop(h).(HammingCandidate)
	}
	slices.SortFunc(candidates, func(a, b HammingCandidate) int {
		if a.Hamming != b.Hamming {
			return a.Hamming - b.Hamming
		}
		if a.Group != b.Group {
			return stringsCompare(a.Group, b.Group)
		}
		return a.ChunkIndex - b.ChunkIndex
	})
	return candidates, true
}

func hammingCandidateBetter(a, b HammingCandidate) bool {
	if a.Hamming != b.Hamming {
		return a.Hamming < b.Hamming
	}
	if a.Group != b.Group {
		return a.Group < b.Group
	}
	return a.ChunkIndex < b.ChunkIndex
}

func hammingCandidateWorse(a, b HammingCandidate) bool {
	if a.Hamming != b.Hamming {
		return a.Hamming > b.Hamming
	}
	if a.Group != b.Group {
		return a.Group > b.Group
	}
	return a.ChunkIndex > b.ChunkIndex
}

func stringsCompare(a, b string) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

// Len returns the number of entries in the index.
func (idx *BinaryIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.entries)
}

// Clear clears the index.
func (idx *BinaryIndex) Clear() {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.entries = nil
	idx.vectors = nil
	idx.medians = nil
}

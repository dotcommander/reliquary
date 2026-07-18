package dedup

import (
	"math"
	"math/bits"
	"slices"
	"strconv"
)

// Detector groups items by content hash to find exact and near duplicates.
type Detector[T any] struct {
	hasher    *ContentHasher
	content   func(T) string
	less      func(a, b T) bool
	hashIndex map[string][]T
}

// Stats is a typed snapshot of a detector index.
type Stats struct {
	TotalFiles        int
	UniqueHashes      int
	DuplicateGroups   int
	DuplicateFiles    int
	DeduplicationRate float64
}

// DuplicateGroup is a group of items that share the same content hash.
type DuplicateGroup[T any] struct {
	Hash  string
	Items []T
}

// NewDetector creates a new duplicate detector for items of type T.
func NewDetector[T any](strategy HashingStrategy, content func(T) string) *Detector[T] {
	return &Detector[T]{
		hasher:    NewContentHasher(strategy),
		content:   content,
		hashIndex: make(map[string][]T),
	}
}

// WithOrdering sets the ordering used to sort items within duplicate groups.
func (d *Detector[T]) WithOrdering(less func(a, b T) bool) *Detector[T] {
	d.less = less
	return d
}

// Index creates a hash index for all items.
func (d *Detector[T]) Index(items []T) {
	d.hashIndex = make(map[string][]T)

	for _, item := range items {
		hash := d.hasher.HashContent(d.content(item))
		d.hashIndex[hash] = append(d.hashIndex[hash], item)
	}
}

// FindDuplicateGroups returns groups of duplicate items with their shared hash.
func (d *Detector[T]) FindDuplicateGroups() []DuplicateGroup[T] {
	var duplicateGroups []DuplicateGroup[T]

	for hash, group := range d.hashIndex {
		if len(group) > 1 {
			group = slices.Clone(group)
			if d.less != nil {
				slices.SortStableFunc(group, func(a, b T) int {
					if d.less(a, b) {
						return -1
					}
					if d.less(b, a) {
						return 1
					}
					return 0
				})
			}

			duplicateGroups = append(duplicateGroups, DuplicateGroup[T]{
				Hash:  hash,
				Items: group,
			})
		}
	}

	// Sort groups by size (largest first); break ties deterministically by
	// the group's first element when an ordering is configured.
	slices.SortStableFunc(duplicateGroups, func(a, b DuplicateGroup[T]) int {
		if len(a.Items) != len(b.Items) {
			if len(a.Items) > len(b.Items) {
				return -1
			}
			return 1
		}
		if d.less != nil {
			if d.less(a.Items[0], b.Items[0]) {
				return -1
			}
			if d.less(b.Items[0], a.Items[0]) {
				return 1
			}
		}
		return 0
	})

	return duplicateGroups
}

// FindDuplicates returns groups of duplicate items.
func (d *Detector[T]) FindDuplicates() [][]T {
	duplicateGroups := d.FindDuplicateGroups()
	groups := make([][]T, 0, len(duplicateGroups))
	for _, group := range duplicateGroups {
		groups = append(groups, group.Items)
	}
	return groups
}

// FindNearDuplicates finds items with similar hashes (for SimHash).
func (d *Detector[T]) FindNearDuplicates(threshold int) [][]T {
	if !d.hasher.SupportsNearDuplicate() {
		return [][]T{}
	}

	var nearDuplicateGroups [][]T
	processed := make(map[string]bool)

	hashes := make([]string, 0, len(d.hashIndex))
	for hash := range d.hashIndex {
		hashes = append(hashes, hash)
	}
	slices.Sort(hashes) // deterministic grouping order

	for i, hash1 := range hashes {
		if processed[hash1] {
			continue
		}

		group := slices.Clone(d.hashIndex[hash1])
		processed[hash1] = true

		for j := i + 1; j < len(hashes); j++ {
			hash2 := hashes[j]
			if processed[hash2] {
				continue
			}

			if hammingDistance(hash1, hash2) <= threshold {
				group = append(group, d.hashIndex[hash2]...)
				processed[hash2] = true
			}
		}

		if len(group) > 1 {
			nearDuplicateGroups = append(nearDuplicateGroups, group)
		}
	}

	return nearDuplicateGroups
}

// hammingDistance returns the bit-level Hamming distance between two
// %016x-encoded 64-bit hashes (range 0–64). Mismatched/unparseable inputs
// return math.MaxInt (treated as maximally distant).
func hammingDistance(hash1, hash2 string) int {
	if len(hash1) != len(hash2) {
		return math.MaxInt
	}
	v1, err1 := strconv.ParseUint(hash1, 16, 64)
	v2, err2 := strconv.ParseUint(hash2, 16, 64)
	if err1 != nil || err2 != nil {
		return math.MaxInt
	}
	return bits.OnesCount64(v1 ^ v2)
}

// Stats returns statistics about the hashing results.
func (d *Detector[T]) Stats() Stats {
	totalFiles := 0
	duplicateGroups := 0
	duplicateFiles := 0

	for _, group := range d.hashIndex {
		totalFiles += len(group)
		if len(group) > 1 {
			duplicateGroups++
			duplicateFiles += len(group)
		}
	}

	var dedupRate float64
	if totalFiles > 0 {
		dedupRate = float64(duplicateFiles) / float64(totalFiles)
	}

	return Stats{
		TotalFiles:        totalFiles,
		UniqueHashes:      len(d.hashIndex),
		DuplicateGroups:   duplicateGroups,
		DuplicateFiles:    duplicateFiles,
		DeduplicationRate: dedupRate,
	}
}

// GetStats returns statistics about the hashing results.
func (d *Detector[T]) GetStats() map[string]any {
	stats := d.Stats()
	return map[string]any{
		"total_files":        stats.TotalFiles,
		"unique_hashes":      stats.UniqueHashes,
		"duplicate_groups":   stats.DuplicateGroups,
		"duplicate_files":    stats.DuplicateFiles,
		"deduplication_rate": stats.DeduplicationRate,
	}
}

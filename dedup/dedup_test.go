package dedup

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHashContentStrategies(t *testing.T) {
	t.Parallel()

	hasher := NewContentHasher(NormalizedHash)

	// normalizedHash lowercases, collapses whitespace, and trims, so these
	// two inputs normalize to the same string "hello world".
	h1 := hasher.HashContent("Hello World")
	h2 := hasher.HashContent("hello   world")

	assert.NotEmpty(t, h1)
	assert.NotEmpty(t, h2)
	assert.Equal(t, h1, h2)
}

func TestDetectorFindDuplicates(t *testing.T) {
	t.Parallel()

	type doc struct{ id, body string }

	d := NewDetector[doc](SimpleHash, func(d doc) string { return d.body })
	d.Index([]doc{
		{id: "a", body: "shared body"},
		{id: "b", body: "shared body"},
		{id: "c", body: "unique body"},
	})

	groups := d.FindDuplicates()
	assert.Len(t, groups, 1)
	assert.Len(t, groups[0], 2)
}

func TestDetectorWithOrdering(t *testing.T) {
	t.Parallel()

	type doc struct{ id, body string }

	d := NewDetector[doc](SimpleHash, func(d doc) string { return d.body }).
		WithOrdering(func(a, b doc) bool { return a.id < b.id })
	d.Index([]doc{
		{id: "b", body: "shared body"},
		{id: "a", body: "shared body"},
		{id: "c", body: "unique body"},
	})

	groups := d.FindDuplicates()
	assert.Len(t, groups, 1)
	assert.Len(t, groups[0], 2)
	assert.Equal(t, "a", groups[0][0].id)
	assert.Equal(t, "b", groups[0][1].id)
}

func TestFindNearDuplicates(t *testing.T) {
	t.Parallel()

	type doc struct{ id, body string }

	// Guard: non-SimHash strategies never yield near-duplicates.
	simple := NewDetector[doc](SimpleHash, func(d doc) string { return d.body })
	simple.Index([]doc{
		{id: "a", body: "the quick brown fox jumps over the lazy dog"},
		{id: "b", body: "the quick brown fox jumps over the lazy dog today"},
	})
	assert.Len(t, simple.FindNearDuplicates(3), 0)

	// SimHash over two near-identical bodies should form one near-dup group.
	sim := NewDetector[doc](SimHash, func(d doc) string { return d.body })
	sim.Index([]doc{
		{id: "a", body: "the quick brown fox jumps over the lazy dog"},
		{id: "b", body: "the quick brown fox jumps over the lazy dog today"},
	})
	near := sim.FindNearDuplicates(5) // bodies differ by 5 bits under current simHash
	assert.Len(t, near, 1)
	assert.Len(t, near[0], 2)
}

func TestGetStats(t *testing.T) {
	t.Parallel()

	type doc struct{ id, body string }

	d := NewDetector[doc](SimpleHash, func(d doc) string { return d.body })
	d.Index([]doc{
		{id: "a", body: "shared body"},
		{id: "b", body: "shared body"},
		{id: "c", body: "unique body"},
	})

	stats := d.GetStats()
	assert.Contains(t, stats, "total_files")
	assert.Contains(t, stats, "unique_hashes")
	assert.Contains(t, stats, "duplicate_groups")
	assert.Contains(t, stats, "duplicate_files")
	assert.Contains(t, stats, "deduplication_rate")
	assert.IsType(t, float64(0), stats["deduplication_rate"])
}

func TestSemanticHashShortLines(t *testing.T) {
	t.Parallel()

	hasher := NewContentHasher(SemanticHash)

	// Two distinct docs composed only of short (<=2 word) plain-text lines
	// must not collide to the empty-string hash.
	h1 := hasher.HashContent("hi there\nok go")
	h2 := hasher.HashContent("bye now\nno stop")

	assert.NotEmpty(t, h1)
	assert.NotEmpty(t, h2)
	assert.NotEqual(t, h1, h2)
}

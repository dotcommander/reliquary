package dedup

import (
	"math"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHammingDistance(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		a, b     string
		expected int
	}{
		{"identical zeros", "0000000000000000", "0000000000000000", 0},
		{"one bit", "0000000000000000", "0000000000000001", 1},
		{"all bits", "0000000000000000", "ffffffffffffffff", 64},
		{"mismatched length", "00", "0000000000000000", math.MaxInt},
		{"non-hex equal length", "zzzzzzzzzzzzzzzz", "0000000000000000", math.MaxInt},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, hammingDistance(tc.a, tc.b))
		})
	}
}

func TestSupportsNearDuplicate(t *testing.T) {
	t.Parallel()

	cases := []struct {
		strategy HashingStrategy
		expected bool
	}{
		{SimpleHash, false},
		{NormalizedHash, false},
		{SemanticHash, false},
		{SimHash, true},
	}

	for _, tc := range cases {
		t.Run(string(tc.strategy), func(t *testing.T) {
			t.Parallel()
			ch := NewContentHasher(tc.strategy)
			assert.Equal(t, tc.expected, ch.SupportsNearDuplicate())
		})
	}
}

func TestFindDuplicatesPerStrategy(t *testing.T) {
	t.Parallel()

	identity := func(s string) string { return s }

	cases := []struct {
		name     string
		strategy HashingStrategy
		items    []string
	}{
		{
			name:     "SimpleHash",
			strategy: SimpleHash,
			// "abc" and "abc" are byte-identical → same SHA256
			items: []string{"abc", "abc", "xyz"},
		},
		{
			name:     "NormalizedHash",
			strategy: NormalizedHash,
			// "Hello World" and "hello   world" both normalize to "hello world"
			items: []string{"Hello World", "hello   world", "different"},
		},
		{
			name:     "SemanticHash",
			strategy: SemanticHash,
			// Both produce identical semantic parts: HEADER:# Title|TEXT:body text here
			items: []string{"# Title\nbody text here", "# Title\nbody text here", "# Other\nnope"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			d := NewDetector[string](tc.strategy, identity)
			d.Index(tc.items)
			groups := d.FindDuplicates()
			assert.Len(t, groups, 1, "expected exactly 1 duplicate group")
			assert.Len(t, groups[0], 2, "expected duplicate group of size 2")
		})
	}
}

func TestGetStatsValues(t *testing.T) {
	t.Parallel()

	identity := func(s string) string { return s }

	t.Run("two duplicates one unique", func(t *testing.T) {
		t.Parallel()
		d := NewDetector[string](SimpleHash, identity)
		d.Index([]string{"x", "x", "y"})
		stats := d.GetStats()

		assert.Equal(t, 3, stats["total_files"])
		assert.Equal(t, 2, stats["unique_hashes"])
		assert.Equal(t, 1, stats["duplicate_groups"])
		assert.Equal(t, 2, stats["duplicate_files"])
		assert.InDelta(t, 2.0/3.0, stats["deduplication_rate"], 1e-9)
	})

	t.Run("empty index", func(t *testing.T) {
		t.Parallel()
		d := NewDetector[string](SimpleHash, identity)
		d.Index([]string{})
		stats := d.GetStats()

		assert.Equal(t, 0, stats["total_files"])
		assert.Equal(t, 0, stats["unique_hashes"])
		assert.Equal(t, 0, stats["duplicate_groups"])
		assert.Equal(t, 0.0, stats["deduplication_rate"])
		assert.Empty(t, d.FindDuplicates())
		assert.Empty(t, d.FindNearDuplicates(3))
	})

	t.Run("single item", func(t *testing.T) {
		t.Parallel()
		d := NewDetector[string](SimpleHash, identity)
		d.Index([]string{"solo"})
		stats := d.GetStats()

		assert.Equal(t, 1, stats["total_files"])
		assert.Equal(t, 1, stats["unique_hashes"])
		assert.Equal(t, 0, stats["duplicate_groups"])
	})
}

func TestTypedStatsMatchesLegacyStats(t *testing.T) {
	t.Parallel()

	d := NewDetector[string](SimpleHash, func(s string) string { return s })
	d.Index([]string{"x", "x", "y"})

	typed := d.Stats()
	legacy := d.GetStats()

	assert.Equal(t, typed.TotalFiles, legacy["total_files"])
	assert.Equal(t, typed.UniqueHashes, legacy["unique_hashes"])
	assert.Equal(t, typed.DuplicateGroups, legacy["duplicate_groups"])
	assert.Equal(t, typed.DuplicateFiles, legacy["duplicate_files"])
	assert.Equal(t, typed.DeduplicationRate, legacy["deduplication_rate"])
}

func TestFindDuplicateGroupsIncludesHashAndClonedItems(t *testing.T) {
	t.Parallel()

	type doc struct{ id, body string }
	d := NewDetector[doc](SimpleHash, func(d doc) string { return d.body }).
		WithOrdering(func(a, b doc) bool { return a.id < b.id })
	d.Index([]doc{
		{id: "b", body: "shared"},
		{id: "a", body: "shared"},
		{id: "c", body: "distinct"},
	})

	groups := d.FindDuplicateGroups()
	require.Len(t, groups, 1)
	assert.Equal(t, d.hasher.HashContent("shared"), groups[0].Hash)
	require.Len(t, groups[0].Items, 2)
	assert.Equal(t, "a", groups[0].Items[0].id)
	assert.Equal(t, "b", groups[0].Items[1].id)

	groups[0].Items[0].id = "mutated"
	hash := d.hasher.HashContent("shared")
	assert.Equal(t, "b", d.hashIndex[hash][0].id, "FindDuplicateGroups must not expose stored buckets")
}

func TestSimHashDeterministic(t *testing.T) {
	t.Parallel()

	ch := NewContentHasher(SimHash)
	input := "the quick brown fox jumps over the lazy dog"

	first := ch.HashContent(input)
	assert.Len(t, first, 16, "simhash must produce exactly 16 hex chars")

	for i := 1; i < 100; i++ {
		got := ch.HashContent(input)
		assert.Equal(t, first, got, "simhash must be deterministic (call %d differed)", i)
	}
}

func TestNormalizedHashUnicodeWhitespace(t *testing.T) {
	t.Parallel()

	hasher := NewContentHasher(NormalizedHash)
	plain := hasher.HashContent("Alpha Beta")

	assert.Equal(t, plain, hasher.HashContent("Alpha\u00a0Beta"), "NBSP must normalize as whitespace")
	assert.Equal(t, plain, hasher.HashContent("Alpha\u2003Beta"), "EM SPACE must normalize as whitespace")
}

func TestSimHashShortAndUnicodeContent(t *testing.T) {
	t.Parallel()

	hasher := NewContentHasher(SimHash)
	zero := "0000000000000000"

	t.Run("one and two character inputs", func(t *testing.T) {
		a := hasher.HashContent("a")
		b := hasher.HashContent("b")
		ab := hasher.HashContent("ab")
		cd := hasher.HashContent("cd")

		assert.NotEqual(t, zero, a)
		assert.NotEqual(t, zero, b)
		assert.NotEqual(t, a, b)
		assert.NotEqual(t, zero, ab)
		assert.NotEqual(t, zero, cd)
		assert.NotEqual(t, ab, cd)
	})

	t.Run("non-Latin input", func(t *testing.T) {
		japanese := hasher.HashContent("日本語")
		cyrillic := hasher.HashContent("русский")

		assert.NotEqual(t, zero, japanese)
		assert.NotEqual(t, zero, cyrillic)
		assert.NotEqual(t, japanese, cyrillic)
	})

	t.Run("Unicode marks and whitespace", func(t *testing.T) {
		withASCIIWhitespace := hasher.HashContent("Cafe\u0301 au lait")
		withUnicodeWhitespace := hasher.HashContent("Cafe\u0301\u00a0au\u2003lait")
		withoutCombiningMark := hasher.HashContent("Cafe au lait")

		assert.Equal(t, withASCIIWhitespace, withUnicodeWhitespace)
		assert.NotEqual(t, withASCIIWhitespace, withoutCombiningMark)
	})

	t.Run("punctuation-only input", func(t *testing.T) {
		exclamation := hasher.HashContent("!!!")
		question := hasher.HashContent("???")

		assert.NotEqual(t, zero, exclamation)
		assert.NotEqual(t, zero, question)
		assert.NotEqual(t, exclamation, question)
	})

	t.Run("empty and whitespace-only input", func(t *testing.T) {
		empty := hasher.HashContent("")

		assert.Equal(t, zero, empty)
		assert.Equal(t, empty, hasher.HashContent(" \t\n"))
		assert.Equal(t, empty, hasher.HashContent("\u00a0\u2003"))
	})

	t.Run("shingle size larger than input", func(t *testing.T) {
		custom := NewContentHasher(SimHash).WithSimHashOptions(10, 64)
		first := custom.HashContent("short")
		second := custom.HashContent("other")

		assert.NotEqual(t, zero, first)
		assert.NotEqual(t, zero, second)
		assert.NotEqual(t, first, second)
	})

	t.Run("all shingle windows filtered", func(t *testing.T) {
		custom := NewContentHasher(SimHash).WithSimHashOptions(2, 64)
		first := custom.HashContent("a b")
		second := custom.HashContent("c d")

		assert.NotEqual(t, zero, first)
		assert.NotEqual(t, zero, second)
		assert.NotEqual(t, first, second)
	})
}

func TestSimHashASCIIGolden(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "76b2571915730268", NewContentHasher(SimHash).HashContent("alpha"))
}

func TestFindNearDuplicatesThreshold(t *testing.T) {
	t.Parallel()

	identity := func(s string) string { return s }
	a := "the quick brown fox jumps over the lazy dog"
	b := a + " today"
	c := "completely unrelated lorem ipsum content string"

	t.Run("near pair threshold 64 groups", func(t *testing.T) {
		t.Parallel()
		d := NewDetector[string](SimHash, identity)
		d.Index([]string{a, b})
		groups := d.FindNearDuplicates(64)
		assert.Len(t, groups, 1, "threshold=64 should merge any pair into one group")
		assert.Len(t, groups[0], 2)
	})

	t.Run("near pair threshold 0 no groups", func(t *testing.T) {
		t.Parallel()
		d := NewDetector[string](SimHash, identity)
		d.Index([]string{a, b})
		// a and b have different simhashes; threshold 0 means exact bit match only
		groups := d.FindNearDuplicates(0)
		assert.Empty(t, groups, "threshold=0 should not group distinct simhashes")
	})

	t.Run("far pair threshold 3 no groups", func(t *testing.T) {
		t.Parallel()
		d := NewDetector[string](SimHash, identity)
		d.Index([]string{a, c})
		groups := d.FindNearDuplicates(3)
		assert.Empty(t, groups, "unrelated documents should not group at threshold=3")
	})

	t.Run("non-simhash gated by SupportsNearDuplicate", func(t *testing.T) {
		t.Parallel()
		d := NewDetector[string](NormalizedHash, identity)
		d.Index([]string{a, b})
		groups := d.FindNearDuplicates(64)
		assert.Empty(t, groups, "non-SimHash strategy must return empty from FindNearDuplicates")
	})
}

func TestEmptyContent(t *testing.T) {
	t.Parallel()

	// All items produce identical hashes for empty content → single group of 3.
	d := NewDetector[int](SimpleHash, func(int) string { return "" })
	d.Index([]int{1, 2, 3})

	groups := d.FindDuplicates()
	assert.Len(t, groups, 1, "all-empty content must collide into one group")
	assert.Len(t, groups[0], 3)

	stats := d.GetStats()
	assert.Equal(t, 1, stats["unique_hashes"])
}

func TestPointerType(t *testing.T) {
	t.Parallel()

	type doc struct{ id, body string }

	d := NewDetector[*doc](SimpleHash, func(p *doc) string { return p.body })
	d.Index([]*doc{
		{id: "a", body: "shared"},
		{id: "b", body: "shared"},
		{id: "c", body: "distinct"},
	})

	groups := d.FindDuplicates()
	assert.Len(t, groups, 1)
	assert.Len(t, groups[0], 2)
}

func TestNilContentFuncPanics(t *testing.T) {
	t.Parallel()

	// Documents current behavior: passing nil as the content func causes a
	// nil function call panic inside Index. No guard exists in the implementation;
	// this test locks in that behavior and will fail if a guard is ever added.
	d := NewDetector[string](SimpleHash, nil)
	assert.Panics(t, func() {
		d.Index([]string{"x"})
	})
}

func TestSimHashConfigurableShingleSize(t *testing.T) {
	t.Parallel()

	input := "the quick brown fox jumps over the lazy dog"

	def := NewContentHasher(SimHash).HashContent(input)
	custom := NewContentHasher(SimHash).WithSimHashOptions(5, 64).HashContent(input)

	assert.Len(t, custom, 16, "simhash output must stay 16 hex chars regardless of shingle size")
	assert.NotEqual(t, def, custom, "a different shingle size should change the simhash for multi-word input")

	again := NewContentHasher(SimHash).WithSimHashOptions(5, 64).HashContent(input)
	assert.Equal(t, custom, again, "simhash with custom shingle size must be deterministic")
}

func TestSimHashConfigurableBitWidth(t *testing.T) {
	t.Parallel()

	input := "the quick brown fox jumps over the lazy dog"

	narrow := NewContentHasher(SimHash).WithSimHashOptions(3, 16).HashContent(input)
	assert.Len(t, narrow, 16, "narrower bit width must still encode as 16 hex chars")

	v, err := strconv.ParseUint(narrow, 16, 64)
	require.NoError(t, err)
	assert.LessOrEqual(t, v, uint64(0xFFFF), "16-bit simhash must not set bits above bit 15")

	clampedHigh := NewContentHasher(SimHash).WithSimHashOptions(3, 128).HashContent(input)
	def := NewContentHasher(SimHash).HashContent(input)
	assert.Equal(t, def, clampedHigh, "bits>64 must clamp to the 64-bit default behavior")

	clampedLow := NewContentHasher(SimHash).WithSimHashOptions(0, 64).HashContent(input)
	assert.Len(t, clampedLow, 16, "shingleSize<1 must clamp to 1 and still produce a valid hash")
}

func TestFindDuplicatesDoesNotMutateIndexBuckets(t *testing.T) {
	t.Parallel()

	type doc struct{ id, body string }
	d := NewDetector[doc](SimpleHash, func(d doc) string { return d.body }).
		WithOrdering(func(a, b doc) bool { return a.id < b.id })
	d.Index([]doc{
		{id: "b", body: "shared"},
		{id: "a", body: "shared"},
	})

	groups := d.FindDuplicates()
	require.Len(t, groups, 1)
	assert.Equal(t, "a", groups[0][0].id)
	assert.Equal(t, "b", groups[0][1].id)

	hash := d.hasher.HashContent("shared")
	require.Len(t, d.hashIndex[hash], 2)
	assert.Equal(t, "b", d.hashIndex[hash][0].id, "FindDuplicates must not sort the stored bucket in place")
	assert.Equal(t, "a", d.hashIndex[hash][1].id, "FindDuplicates must preserve indexed order")
}

func TestFindDuplicatesConcurrentReadsWithOrdering(t *testing.T) {
	t.Parallel()

	type doc struct{ id, body string }

	items := make([]doc, 0, 200)
	for i := 0; i < 200; i++ {
		items = append(items, doc{id: strconv.Itoa(200 - i), body: "shared"})
	}

	d := NewDetector[doc](SimpleHash, func(d doc) string { return d.body }).
		WithOrdering(func(a, b doc) bool { return a.id < b.id })
	d.Index(items)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				groups := d.FindDuplicates()
				if len(groups) != 1 || len(groups[0]) != len(items) {
					t.Errorf("unexpected duplicate groups: got %d groups", len(groups))
					return
				}
			}
		}()
	}
	wg.Wait()
}

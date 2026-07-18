package textutil

import (
	"cmp"
	"slices"
	"strings"
)

type KeywordOptions struct {
	Limit     int
	MinLength int
	MinCount  int
	StopWords map[string]struct{}
	Include   func(string) bool
}

var defaultStopWords = map[string]struct{}{
	"the":     {},
	"and":     {},
	"or":      {},
	"but":     {},
	"in":      {},
	"on":      {},
	"at":      {},
	"to":      {},
	"for":     {},
	"of":      {},
	"with":    {},
	"by":      {},
	"is":      {},
	"are":     {},
	"was":     {},
	"were":    {},
	"be":      {},
	"been":    {},
	"have":    {},
	"has":     {},
	"had":     {},
	"do":      {},
	"does":    {},
	"did":     {},
	"will":    {},
	"would":   {},
	"could":   {},
	"should":  {},
	"may":     {},
	"might":   {},
	"can":     {},
	"this":    {},
	"that":    {},
	"these":   {},
	"those":   {},
	"a":       {},
	"an":      {},
	"as":      {},
	"if":      {},
	"then":    {},
	"than":    {},
	"so":      {},
	"very":    {},
	"just":    {},
	"now":     {},
	"here":    {},
	"there":   {},
	"when":    {},
	"where":   {},
	"why":     {},
	"how":     {},
	"what":    {},
	"who":     {},
	"which":   {},
	"whom":    {},
	"whose":   {},
	"from":    {},
	"into":    {},
	"about":   {},
	"through": {},
	"during":  {},
	"before":  {},
	"after":   {},
	"above":   {},
	"below":   {},
	"between": {},
	"among":   {},
	"under":   {},
	"over":    {},
}

// DefaultStopWords preserves the historical map-shaped API for callers that
// index, range, or pass the default stop-word set.
//
// Package behavior reads an internal immutable default set, so mutating this
// exported compatibility snapshot never changes IsStopWord or ExtractKeywords.
// Use DefaultStopWordsCopy, KeywordOptions.StopWords, or KeywordOptions.Include
// for per-call filtering.
var DefaultStopWords = DefaultStopWordsCopy()

// DefaultStopWordsCopy returns a copy of the package default stop-word set.
func DefaultStopWordsCopy() map[string]struct{} {
	words := make(map[string]struct{}, len(defaultStopWords))
	for word, marker := range defaultStopWords {
		words[word] = marker
	}
	return words
}

func ExtractKeywords(texts []string, options KeywordOptions) []string {
	limit := options.Limit
	if limit <= 0 {
		limit = 10
	}
	minLength := options.MinLength
	if minLength <= 0 {
		minLength = 4
	}

	counts := make(map[string]int)
	for _, text := range texts {
		for _, raw := range strings.Fields(strings.ToLower(text)) {
			token := NormalizeKeywordToken(raw)
			if token == "" || len(token) < minLength || isStopWord(token, options.StopWords) {
				continue
			}
			if options.Include != nil && !options.Include(token) {
				continue
			}
			counts[token]++
		}
	}

	return topCountedValues(counts, limit, options.MinCount)
}

// MostFrequentValue returns the most frequently occurring trimmed, non-empty
// value in values, provided it appears at least minCount times; otherwise it
// returns fallback. When multiple values share the top frequency, the
// alphabetically smallest value wins (deterministic tie-break via topCountedValues).
func MostFrequentValue(values []string, minCount int, fallback string) string {
	counts := make(map[string]int)
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		counts[trimmed]++
	}

	valuesByCount := topCountedValues(counts, 1, minCount)
	if len(valuesByCount) == 0 {
		return fallback
	}
	return valuesByCount[0]
}

func DetectTheme(content string, themeKeywords map[string][]string, fallback string) string {
	wordCounts := make(map[string]int)
	for _, field := range strings.Fields(strings.ToLower(content)) {
		wordCounts[field]++
	}

	bestTheme := fallback
	bestScore := 0

	themes := make([]string, 0, len(themeKeywords))
	for theme := range themeKeywords {
		themes = append(themes, theme)
	}
	slices.Sort(themes)

	for _, theme := range themes {
		score := 0
		for _, keyword := range themeKeywords[theme] {
			score += wordCounts[strings.ToLower(keyword)]
		}
		if score > bestScore {
			bestScore = score
			bestTheme = theme
		}
	}

	return bestTheme
}

func NormalizeKeywordToken(token string) string {
	return strings.Trim(token, ".,;:()[]{}\"'`!?<>")
}

func IsStopWord(word string) bool {
	_, exists := defaultStopWords[word]
	return exists
}

func isStopWord(word string, extra map[string]struct{}) bool {
	if IsStopWord(word) {
		return true
	}
	_, exists := extra[word]
	return exists
}

func topCountedValues(counts map[string]int, limit, minCount int) []string {
	type scoredValue struct {
		value string
		count int
	}

	scored := make([]scoredValue, 0, len(counts))
	for value, count := range counts {
		if count >= minCount {
			scored = append(scored, scoredValue{value: value, count: count})
		}
	}

	slices.SortFunc(scored, func(a, b scoredValue) int {
		if a.count == b.count {
			return cmp.Compare(a.value, b.value)
		}
		return cmp.Compare(b.count, a.count)
	})

	if len(scored) > limit {
		scored = scored[:limit]
	}

	result := make([]string, 0, len(scored))
	for _, item := range scored {
		result = append(result, item.value)
	}
	return result
}

package vectors

import (
	"regexp"
	"strings"
	"unicode"
)

var (
	codeBlockRE  = regexp.MustCompile("(?s)```[\\s\\S]*?```")
	tildeBlockRE = regexp.MustCompile(`(?s)~.*?~`)
	mathBlockRE  = regexp.MustCompile(`(?s)\\$.*?\\$`)
	urlRE        = regexp.MustCompile(`https?://[^\\s]+`)
	wordRegex    = regexp.MustCompile(`[a-zA-Z0-9]{2,}`)
)

// stopWords is a read-only set of common words excluded from extraction.
// Do not mutate (shared across calls).
var stopWords = map[string]struct{}{
	"the": {}, "a": {}, "an": {}, "and": {}, "or": {}, "but": {}, "in": {}, "on": {}, "at": {}, "to": {},
	"for": {}, "of": {}, "with": {}, "by": {}, "from": {}, "as": {}, "is": {}, "was": {}, "are": {},
	"been": {}, "be": {}, "have": {}, "has": {}, "had": {}, "do": {}, "does": {}, "did": {}, "will": {},
	"would": {}, "could": {}, "should": {}, "may": {}, "might": {}, "must": {}, "can": {},
	"this": {}, "that": {}, "these": {}, "those": {}, "it": {}, "its": {}, "they": {}, "them": {},
	"what": {}, "which": {}, "who": {}, "when": {}, "where": {}, "why": {}, "how": {},
	"all": {}, "each": {}, "every": {}, "both": {}, "few": {}, "more": {}, "most": {},
	"some": {}, "such": {}, "only": {}, "own": {}, "same": {}, "so": {}, "than": {}, "too": {},
	"very": {}, "just": {}, "also": {}, "now": {}, "then": {}, "here": {}, "there": {},
	"up": {}, "down": {}, "out": {}, "over": {}, "under": {}, "again": {},
	"further": {}, "once": {}, "not": {}, "no": {}, "yes": {}, "true": {}, "false": {},
}

// JaccardWords computes Jaccard similarity between meaningful words in two texts.
// Returns 0 for empty inputs.
func JaccardWords(text1, text2 string) float64 {
	words1 := extractWords(text1)
	words2 := extractWords(text2)

	if len(words1) == 0 && len(words2) == 0 {
		return 0.0
	}
	if len(words1) == 0 || len(words2) == 0 {
		return 0.0
	}

	set1 := make(map[string]struct{}, len(words1))
	set2 := make(map[string]struct{}, len(words2))

	for _, w := range words1 {
		set1[strings.ToLower(w)] = struct{}{}
	}
	for _, w := range words2 {
		set2[strings.ToLower(w)] = struct{}{}
	}

	intersection := 0
	for word := range set1 {
		if _, ok := set2[word]; ok {
			intersection++
		}
	}

	union := len(set1) + len(set2) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// extractWords extracts meaningful words from markdown text.
// Excludes code blocks, URLs, and common punctuation.
func extractWords(text string) []string {
	text = codeBlockRE.ReplaceAllString(text, " ")
	text = tildeBlockRE.ReplaceAllString(text, " ")
	text = mathBlockRE.ReplaceAllString(text, " ")
	text = urlRE.ReplaceAllString(text, "")

	raw := wordRegex.FindAllString(text, -1)
	words := make([]string, 0, len(raw))
	for _, word := range raw {
		if isStopWord(word) {
			continue
		}
		words = append(words, word)
	}
	return words
}

func isStopWord(word string) bool {
	lower := strings.ToLower(word)
	if len(word) <= 1 {
		return true
	}
	if isNumeric(word) {
		return true
	}
	_, ok := stopWords[lower]
	return ok
}

func isNumeric(s string) bool {
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

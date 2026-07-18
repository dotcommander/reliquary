package retrieval

import (
	"strings"
	"unicode"
)

func keywordOverlap(query, content string) float64 {
	return overlapRatio(tokenize(query), tokenize(content))
}

func FilenameOverlap(filename, categoryName string) float64 {
	return overlapRatio(tokenize(filename), tokenize(categoryName))
}

func overlapRatio(tokens, reference []string) float64 {
	if len(tokens) == 0 || len(reference) == 0 {
		return 0
	}
	referenceSet := make(map[string]bool, len(reference))
	for _, token := range reference {
		referenceSet[token] = true
	}
	matches := 0
	for _, token := range tokens {
		if referenceSet[token] {
			matches++
		}
	}
	return float64(matches) / float64(len(tokens))
}

func tokenize(text string) []string {
	text = strings.ToLower(text)
	tokens := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	result := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if len(t) > 2 && !isStopword(t) {
			result = append(result, t)
		}
	}
	return result
}

func isStopword(word string) bool {
	return stopwords[word]
}

var stopwords = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true,
	"but": true, "in": true, "on": true, "at": true, "to": true,
	"for": true, "of": true, "with": true, "by": true, "from": true,
	"is": true, "are": true, "was": true, "were": true, "be": true,
	"been": true, "being": true, "have": true, "has": true, "had": true,
	"do": true, "does": true, "did": true, "will": true, "would": true,
	"could": true, "should": true, "may": true, "might": true, "must": true,
	"this": true, "that": true, "these": true, "those": true,
	"it": true, "its": true, "they": true, "them": true, "their": true,
}

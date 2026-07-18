package lexical

import (
	"strings"
	"unicode"
)

// Token is one normalized lexical token and its source location.
type Token struct {
	// Text is the normalized term text consumed by query and scoring helpers.
	Text string
	// StartByte is the inclusive byte offset in the source text.
	StartByte int
	// EndByte is the exclusive byte offset in the source text.
	EndByte int
	// Position is the deterministic token position after filtering.
	Position int
}

// Analyzer turns source text into deterministic lexical tokens.
type Analyzer interface {
	Analyze(text string) []Token
}

// AnalyzerFunc adapts a function into an Analyzer.
type AnalyzerFunc func(text string) []Token

// Analyze calls f(text).
func (f AnalyzerFunc) Analyze(text string) []Token {
	if f == nil {
		return nil
	}
	return f(text)
}

// AnalyzerOptions configures the default analyzer.
type AnalyzerOptions struct {
	// MinTokenLength drops tokens shorter than this many runes. Values <= 0 mean 1.
	MinTokenLength int
	// StopWords drops exact lowercase token matches. Nil means no stop words.
	StopWords map[string]struct{}
}

type defaultAnalyzer struct {
	options AnalyzerOptions
}

// NewAnalyzer returns the package default analyzer configured by options.
func NewAnalyzer(options AnalyzerOptions) Analyzer {
	return defaultAnalyzer{options: options}
}

// Analyze lowercases Unicode letter/number runs and splits on punctuation,
// whitespace, symbols, and marks.
func (a defaultAnalyzer) Analyze(text string) []Token {
	minLength := a.options.MinTokenLength
	if minLength <= 0 {
		minLength = 1
	}

	var tokens []Token
	var b strings.Builder
	start := -1
	for offset, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			if start < 0 {
				start = offset
			}
			b.WriteRune(unicode.ToLower(r))
			continue
		}
		tokens = appendAnalyzedToken(tokens, b.String(), start, offset, minLength, a.options.StopWords)
		b.Reset()
		start = -1
	}
	tokens = appendAnalyzedToken(tokens, b.String(), start, len(text), minLength, a.options.StopWords)
	if tokens == nil {
		return []Token{}
	}
	return tokens
}

func appendAnalyzedToken(tokens []Token, text string, start, end, minLength int, stopWords map[string]struct{}) []Token {
	if text == "" {
		return tokens
	}
	if runeLen(text) < minLength {
		return tokens
	}
	if _, stop := stopWords[text]; stop {
		return tokens
	}
	return append(tokens, Token{
		Text:      text,
		StartByte: start,
		EndByte:   end,
		Position:  len(tokens),
	})
}

func runeLen(s string) int {
	n := 0
	for range s {
		n++
	}
	return n
}

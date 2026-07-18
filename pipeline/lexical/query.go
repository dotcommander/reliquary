package lexical

// Query is a normalized lexical query.
type Query struct {
	// Tokens preserves deterministic analyzer order, including repeated terms.
	Tokens []Token
	// Terms contains per-term counts for Tokens.
	Terms map[string]int
}

// NormalizeQuery analyzes text and counts query terms. Empty or stopword-only
// input returns an empty query, not an error. The analyzer argument may be nil
// or implement Analyzer.
func NormalizeQuery(text string, analyzer any) Query {
	tokens := analyze(text, analyzer)
	return Query{
		Tokens: tokens,
		Terms:  TermCounts(TokenTexts(tokens)),
	}
}

// Empty reports whether the query contains no lexical terms.
func (q Query) Empty() bool {
	return len(q.Tokens) == 0
}

// TokenTexts returns token text in rank/analyzer order.
func TokenTexts(tokens []Token) []string {
	texts := make([]string, len(tokens))
	for i, token := range tokens {
		texts[i] = token.Text
	}
	return texts
}

// TermCounts counts normalized term occurrences.
func TermCounts(terms []string) map[string]int {
	counts := make(map[string]int)
	for _, term := range terms {
		counts[term]++
	}
	return counts
}

func analyze(text string, analyzer any) []Token {
	switch a := analyzer.(type) {
	case nil:
		return NewAnalyzer(AnalyzerOptions{}).Analyze(text)
	case Analyzer:
		return a.Analyze(text)
	case Tokenizer:
		terms := a.Tokens(text)
		tokens := make([]Token, len(terms))
		for i, term := range terms {
			tokens[i] = Token{Text: term, Position: i}
		}
		return tokens
	default:
		return NewAnalyzer(AnalyzerOptions{}).Analyze(text)
	}
}

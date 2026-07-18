package lexical

// Tokenizer is the pre-analyzer token contract. New code should prefer
// Analyzer when byte offsets or positions matter.
type Tokenizer interface {
	Tokens(text string) []string
}

// TokenOptions configures DefaultTokenizer.
type TokenOptions struct {
	// MinLength drops tokens shorter than this many runes. Values <= 0 mean 1.
	MinLength int
	// StopWords drops exact lowercase token matches. Nil means no stop words.
	StopWords map[string]struct{}
}

// DefaultTokenizer returns normalized token text without offsets.
type DefaultTokenizer struct {
	Options TokenOptions
}

// NewTokenizer returns a DefaultTokenizer with the supplied options.
func NewTokenizer(options TokenOptions) DefaultTokenizer {
	return DefaultTokenizer{Options: options}
}

// Tokens returns lowercase Unicode letter/number tokens in input order.
func (t DefaultTokenizer) Tokens(text string) []string {
	tokens := NewAnalyzer(AnalyzerOptions{
		MinTokenLength: t.Options.MinLength,
		StopWords:      t.Options.StopWords,
	}).Analyze(text)
	return TokenTexts(tokens)
}

// CollectionStats is the former name for CorpusStats.
type CollectionStats = CorpusStats

// NewCollectionStats builds corpus statistics from per-document stats.
func NewCollectionStats(docs []DocumentStats) CorpusStats {
	return NewCorpusStats(docs)
}

// BM25 is the former name for BM25Score.
func BM25(query Query, doc DocumentStats, corpus CorpusStats, params BM25Params) float64 {
	return BM25Score(query, doc, corpus, params)
}

// Ranked is the former name for Candidate.
type Ranked = Candidate

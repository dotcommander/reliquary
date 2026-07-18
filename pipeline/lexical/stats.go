package lexical

// DocumentStats contains local lexical statistics for one document.
type DocumentStats struct {
	TermFrequency map[string]int
	Length        int
}

// CorpusStats contains corpus-level statistics required by BM25.
type CorpusStats struct {
	DocumentCount         int
	AverageDocumentLength float64
	DocumentFrequency     map[string]int
}

// NewDocumentStats builds document statistics from already-normalized terms.
func NewDocumentStats(terms []string) DocumentStats {
	return DocumentStats{
		TermFrequency: TermCounts(terms),
		Length:        len(terms),
	}
}

// NewDocumentStatsFromTokens builds document statistics from analyzed tokens.
func NewDocumentStatsFromTokens(tokens []Token) DocumentStats {
	return NewDocumentStats(TokenTexts(tokens))
}

// NewCorpusStats builds corpus statistics from per-document stats.
func NewCorpusStats(docs []DocumentStats) CorpusStats {
	if len(docs) == 0 {
		return CorpusStats{DocumentFrequency: map[string]int{}}
	}

	df := make(map[string]int)
	var totalLength int
	for _, doc := range docs {
		totalLength += doc.Length
		for term, freq := range doc.TermFrequency {
			if freq > 0 {
				df[term]++
			}
		}
	}

	return CorpusStats{
		DocumentCount:         len(docs),
		AverageDocumentLength: float64(totalLength) / float64(len(docs)),
		DocumentFrequency:     df,
	}
}

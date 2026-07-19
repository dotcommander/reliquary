package lexical

import (
	"reflect"
	"testing"
)

func TestCompatTokenizer(t *testing.T) {
	t.Parallel()

	tok := NewTokenizer(TokenOptions{
		MinLength: 2,
		StopWords: map[string]struct{}{"the": {}},
	})

	tokens := tok.Tokens("The quick brown fox 123 a")
	want := []string{"quick", "brown", "fox", "123"}
	if !reflect.DeepEqual(tokens, want) {
		t.Fatalf("Tokens() = %v, want %v", tokens, want)
	}
}

func TestCompatCollectionStatsAndBM25(t *testing.T) {
	t.Parallel()

	analyzer := NewAnalyzer(AnalyzerOptions{})
	docA := NewDocumentStatsFromTokens(analyzer.Analyze("alpha beta"))
	docB := NewDocumentStatsFromTokens(analyzer.Analyze("alpha gamma"))
	corpus := NewCollectionStats([]DocumentStats{docA, docB})

	if corpus.DocumentCount != 2 {
		t.Fatalf("DocumentCount = %d, want 2", corpus.DocumentCount)
	}

	query := NormalizeQuery("alpha", analyzer)
	score := BM25(query, docA, corpus, DefaultBM25Params())
	if score <= 0 {
		t.Fatalf("BM25 score = %v, want > 0", score)
	}
}

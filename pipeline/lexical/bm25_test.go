package lexical

import (
	"math"
	"testing"
)

func TestBM25Score_SmallCorpusFixture(t *testing.T) {
	t.Parallel()

	analyzer := NewAnalyzer(AnalyzerOptions{})
	doc1 := NewDocumentStatsFromTokens(analyzer.Analyze("alpha alpha beta"))
	doc2 := NewDocumentStatsFromTokens(analyzer.Analyze("alpha gamma"))
	corpus := NewCorpusStats([]DocumentStats{doc1, doc2})
	query := NormalizeQuery("alpha beta", analyzer)

	got := BM25Score(query, doc1, corpus, DefaultBM25Params())

	const totalDocs = 2.0       // N: corpus has two documents.
	alphaDF, betaDF := 2.0, 1.0 // document frequency of each query term.
	alphaIDF := math.Log(1 + (totalDocs-alphaDF+0.5)/(alphaDF+0.5))
	betaIDF := math.Log(1 + (totalDocs-betaDF+0.5)/(betaDF+0.5))
	alphaDenom := 2.0 + 1.2*(1-0.75+0.75*3.0/2.5)
	betaDenom := 1.0 + 1.2*(1-0.75+0.75*3.0/2.5)
	want := alphaIDF*((2.0*(1.2+1))/alphaDenom) + betaIDF*((1.0*(1.2+1))/betaDenom)

	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("BM25Score() = %.15f, want %.15f", got, want)
	}
}

func TestBM25Score_MissingTermsAndInvalidStatsScoreZero(t *testing.T) {
	t.Parallel()

	analyzer := NewAnalyzer(AnalyzerOptions{})
	doc := NewDocumentStatsFromTokens(analyzer.Analyze("alpha beta"))
	corpus := NewCorpusStats([]DocumentStats{doc})

	if got := BM25Score(NormalizeQuery("gamma", analyzer), doc, corpus, DefaultBM25Params()); got != 0 {
		t.Fatalf("BM25Score(missing term) = %f, want 0", got)
	}
	if got := BM25Score(NormalizeQuery("alpha", analyzer), doc, CorpusStats{}, DefaultBM25Params()); got != 0 {
		t.Fatalf("BM25Score(invalid corpus) = %f, want 0", got)
	}
	if got := BM25Score(Query{}, doc, corpus, DefaultBM25Params()); got != 0 {
		t.Fatalf("BM25Score(empty query) = %f, want 0", got)
	}
}

func TestBM25Score_ZeroValueParamsUseDefaults(t *testing.T) {
	t.Parallel()

	analyzer := NewAnalyzer(AnalyzerOptions{})
	doc := NewDocumentStatsFromTokens(analyzer.Analyze("alpha alpha beta"))
	corpus := NewCorpusStats([]DocumentStats{doc})
	query := NormalizeQuery("alpha", analyzer)

	got := BM25Score(query, doc, corpus, BM25Params{})
	want := BM25Score(query, doc, corpus, DefaultBM25Params())

	if got != want {
		t.Fatalf("BM25Score zero-value params = %f, want default params score %f", got, want)
	}
}

func TestNewCorpusStats_Empty(t *testing.T) {
	t.Parallel()

	got := NewCorpusStats(nil)
	if got.DocumentCount != 0 {
		t.Fatalf("DocumentCount = %d, want 0", got.DocumentCount)
	}
	if got.DocumentFrequency == nil {
		t.Fatal("DocumentFrequency is nil, want non-nil empty map")
	}
}

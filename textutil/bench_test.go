package textutil

import "testing"

// sinkStrings / sinkFloat prevent the compiler from eliding benchmark results.
var sinkStrings []string
var sinkFloat float64
var sinkMatch AliasMatch

func BenchmarkExtractKeywords(b *testing.B) {
	b.ReportAllocs()
	texts := []string{
		"The quick brown fox jumps over the lazy dog. " +
			"Concurrency in Go is achieved through goroutines and channels. " +
			"Keywords extraction filters stop words and scores term frequency across documents.",
	}
	opts := KeywordOptions{Limit: 10, MinCount: 1}
	var r []string
	for i := 0; i < b.N; i++ {
		r = ExtractKeywords(texts, opts)
	}
	sinkStrings = r
}

func BenchmarkStringSimilarity(b *testing.B) {
	b.ReportAllocs()
	var r float64
	for i := 0; i < b.N; i++ {
		r = StringSimilarity("retrieval", "retreival")
	}
	sinkFloat = r
}

func BenchmarkAliasQueryScore(b *testing.B) {
	b.ReportAllocs()
	aliases := []string{"Alex Q. Example", "Alex Example"}
	var r AliasMatch
	for i := 0; i < b.N; i++ {
		r = AliasQueryScore("Alex Quinn Example", "Alex Quinn Example", aliases, IsStopWord)
	}
	sinkMatch = r
}

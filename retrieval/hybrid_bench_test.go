package retrieval

import (
	"fmt"
	"math/rand"
	"testing"
)

// benchResults returns deterministic scored-candidate fixtures. Embeddings are
// unnormalized so the benchmark exercises the scaled-accumulation norm path.
func benchResults(n, dim int, r *rand.Rand) []*Result {
	results := make([]*Result, n)
	for i := range results {
		embedding := make([]float64, dim)
		for d := range embedding {
			embedding[d] = r.NormFloat64()
		}
		results[i] = &Result{
			ID:        fmt.Sprintf("result-%06d", i),
			Content:   "fixed benchmark content chunk with several plain scoring words",
			Filename:  fmt.Sprintf("file-%06d.md", i),
			Embedding: embedding,
		}
	}
	return results
}

func benchQuery(dim int, r *rand.Rand) []float64 {
	query := make([]float64, dim)
	for d := range query {
		query[d] = r.NormFloat64()
	}
	return query
}

// BenchmarkScorerRerank_768_10k measures one full hybrid scoring pass (cosine
// plus tokenization, calibration, and sort) over 10k candidates at 768 dims.
func BenchmarkScorerRerank_768_10k(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	query := benchQuery(768, r)
	results := benchResults(10_000, 768, r)
	scorer := NewScorer(DefaultWeights())
	b.ResetTimer()
	for b.Loop() {
		scorer.Rerank(query, "benchmark hybrid scoring query", results)
	}
}

// BenchmarkDiversify_768_1k_K10 measures MMR selection of 10 results from 1k
// candidates at 768 dims with lambda 0.5.
func BenchmarkDiversify_768_1k_K10(b *testing.B) {
	r := rand.New(rand.NewSource(2))
	results := benchResults(1_000, 768, r)
	for i, result := range results {
		result.CombinedScore = float64(len(results) - i)
	}
	b.ResetTimer()
	for b.Loop() {
		Diversify(results, 10, 0.5)
	}
}

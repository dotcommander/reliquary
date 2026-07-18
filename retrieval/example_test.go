package retrieval_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/retrieval"
)

func ExampleScorer_Rerank() {
	scorer := retrieval.NewScorer(retrieval.DefaultWeights())

	// Example data (embeddings must match dimensional lengths)
	queryEmb := []float64{0.1, 0.2, 0.3}
	results := []*retrieval.Result{
		{
			ID:        "doc1",
			Content:   "machine learning algorithms and modeling",
			Filename:  "machine-learning.md",
			Embedding: []float64{0.12, 0.18, 0.31},
		},
		{
			ID:        "doc2",
			Content:   "italian pizza baking recipe",
			Filename:  "cooking.md",
			Embedding: []float64{0.85, 0.05, 0.10},
		},
	}

	ranked := scorer.Rerank(queryEmb, "machine learning", results)

	for _, r := range ranked {
		fmt.Printf("- %s: Score %.4f (Embedding: %.4f, Keyword: %.4f)\n", r.ID, r.CombinedScore, r.EmbeddingScore, r.KeywordScore)
	}
	// Output:
	// - doc1: Score 1.0000 (Embedding: 1.0000, Keyword: 1.0000)
	// - doc2: Score 0.0000 (Embedding: 0.0000, Keyword: 0.0000)
}

func ExampleMMR() {
	// Diversify items to avoid returning multiple identical documents
	items := []retrieval.MMRItem{
		{
			ID:        "doc1",
			Score:     0.95,
			Embedding: []float64{0.1, 0.2, 0.3},
		},
		{
			ID:        "doc2", // Highly redundant with doc1 (identical embedding)
			Score:     0.92,
			Embedding: []float64{0.1, 0.2, 0.3},
		},
		{
			ID:        "doc3", // Lower score but different topic
			Score:     0.75,
			Embedding: []float64{0.9, 0.1, 0.0},
		},
	}

	// k = 2, lambda = 0.5 (equal balance of relevance and diversity)
	diversified := retrieval.MMR(items, 2, 0.5)

	for _, item := range diversified {
		fmt.Printf("- %s (Score: %.2f)\n", item.ID, item.Score)
	}
	// Output:
	// - doc1 (Score: 0.95)
	// - doc3 (Score: 0.75)
}

func ExampleEvaluatePlan() {
	query := retrieval.EvalQuery{
		ID:       "q1",
		Relevant: map[string]float64{"doc1": 1, "doc3": 1},
	}
	plan := retrieval.Plan{
		ID:     "hybrid",
		Fusion: retrieval.FusionModeRRF,
		Sources: []retrieval.CandidateSource{
			{ID: "lexical", ScoreSpace: "local_bm25", Limit: 2},
			{ID: "vector", ScoreSpace: "cosine", Limit: 2},
		},
	}
	sources := []retrieval.SourceReport{
		{Source: plan.Sources[0], Results: []retrieval.RankedResult{{ID: "doc1"}, {ID: "doc2"}}},
		{Source: plan.Sources[1], Results: []retrieval.RankedResult{{ID: "doc3"}, {ID: "doc4"}}},
	}
	layers := retrieval.LayeredResults{
		Candidates: []retrieval.RankedResult{{ID: "doc1"}, {ID: "doc2"}, {ID: "doc3"}, {ID: "doc4"}},
		Final:      []retrieval.RankedResult{{ID: "doc1"}, {ID: "doc3"}},
	}

	run := retrieval.EvaluatePlan(query, plan, layers, sources, 2)
	fmt.Println(run.Report.CandidateRecall, run.Sources[0].CandidateRecall)
	// Output: 1 0.5
}

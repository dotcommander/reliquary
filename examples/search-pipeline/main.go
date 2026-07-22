package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/retrieval"
)

type termReranker struct {
	term string
}

func (r termReranker) Rerank(_ context.Context, _ string, candidates []*retrieval.Result) ([]float64, error) {
	scores := make([]float64, len(candidates))
	for i, candidate := range candidates {
		scores[i] = 0.1
		if strings.Contains(strings.ToLower(candidate.Content), r.term) {
			scores[i] = 0.9
		}
	}
	return scores, nil
}

func main() {
	ctx := context.Background()
	app := reliquary.Quickstart()

	_, err := app.Ingest(ctx,
		document.Document{ID: "go", Text: "Go uses a concurrent garbage collector."},
		document.Document{ID: "rust", Text: "Rust manages memory through ownership and borrowing."},
		document.Document{ID: "pasta", Text: "Pasta dough rests before rolling."},
	)
	must(err)

	options := []reliquary.SearchOption{
		reliquary.CandidateLimit(10),
		reliquary.WithRRF(60),
		reliquary.WithReranker(termReranker{term: "concurrent"}),
		reliquary.TopK(2),
		reliquary.WithMMR(0.5),
		reliquary.WithExplain(),
	}

	rows, err := app.SearchBatch(ctx, []string{"garbage collector", "", "memory ownership"}, options...)
	must(err)
	fmt.Printf("rows=%d blank=%t first-rank=%d\n", len(rows), rows[1] == nil, rows[0][0].Explain.FinalRank)

	hits, err := app.Search(ctx, "garbage collector", options...)
	must(err)
	for _, hit := range hits {
		explain := hit.Explain
		fmt.Printf(
			"rank=%d id=%s hybrid=%d rrf=%d reranker=%d mmr=%.3f score=%.3f\n",
			explain.FinalRank,
			hit.DocumentID,
			explain.HybridRank,
			explain.RRF.FusedRank,
			explain.Reranker.Rank,
			explain.MMR.SelectionScore,
			hit.CombinedScore,
		)
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

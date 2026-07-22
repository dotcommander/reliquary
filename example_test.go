package reliquary_test

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/retrieval"
)

type exampleReranker struct{}

func (exampleReranker) Rerank(_ context.Context, _ string, candidates []*retrieval.Result) ([]float64, error) {
	scores := make([]float64, len(candidates))
	for i := range scores {
		scores[i] = 1 - float64(i)/float64(len(scores))
	}
	return scores, nil
}

func ExampleQuickstart() {
	app := reliquary.Quickstart()
	ctx := context.Background()
	_, _ = app.Ingest(ctx, document.Document{
		ID:   "doc-1",
		Text: "Go uses a concurrent garbage collector.",
	})

	hits, _ := app.Search(ctx, "garbage collector", reliquary.TopK(1))
	fmt.Println(len(hits), hits[0].ID != "")
	// Output: 1 true
}

func ExampleApp_SearchBatch() {
	app := reliquary.Quickstart()
	ctx := context.Background()
	_, _ = app.Ingest(ctx, document.Document{ID: "doc-1", Text: "Go uses a concurrent garbage collector."})

	rows, _ := app.SearchBatch(ctx, []string{"garbage collector", "", "Go memory"}, reliquary.TopK(1))
	fmt.Println(len(rows), len(rows[0]), rows[1] == nil, len(rows[2]))
	// Output: 3 1 true 1
}

func ExampleWithReranker() {
	app := reliquary.Quickstart()
	ctx := context.Background()
	bgeReranker := exampleReranker{}

	hits, err := app.Search(
		ctx,
		"garbage collector",
		reliquary.CandidateLimit(50),
		reliquary.WithReranker(bgeReranker),
		reliquary.TopK(5),
	)
	_, _ = hits, err
}

func ExampleWithRRF() {
	app := reliquary.Quickstart()
	ctx := context.Background()
	query := "garbage collector"

	hits, err := app.Search(ctx, query,
		reliquary.CandidateLimit(50),
		reliquary.WithRRF(60),
		reliquary.TopK(5),
	)
	_, _ = hits, err
}

func ExampleWithExplain() {
	app := reliquary.Quickstart()
	ctx := context.Background()
	_, _ = app.Ingest(ctx, document.Document{ID: "doc-1", Text: "Go uses a concurrent garbage collector."})

	hits, _ := app.Search(ctx, "garbage collector", reliquary.TopK(1), reliquary.WithExplain())
	explanation := hits[0].Explain
	fmt.Println(explanation.FinalRank, explanation.Hybrid.Raw.Keyword > 0)
	// Output: 1 true
}

// Command quickstart-rag is the short, copyable retrieval path.
//
// It shows the intended happy path using the high-level reliquary facade:
//
//	documents -> Ingest -> Search (ranked + diversified results)
//
// The lower-level examples (rag-ingest-retrieve, chunking-strategies) still
// show each package in detail. This example is deliberately compact so a new
// caller can see how Reliquary fits together.
//
// Run from the repo root:
//
//	GOWORK=off go run ./examples/quickstart-rag
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
)

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	app := reliquary.Quickstart()

	if _, err := app.Ingest(ctx, docs()...); err != nil {
		return err
	}

	query := "how does Go reclaim memory"
	top, err := app.Search(ctx, query, reliquary.TopK(1), reliquary.WithMMR(0.5))
	if err != nil {
		return err
	}

	fmt.Printf("Query: %s\n", query)
	for _, result := range top {
		fmt.Printf("- %s %.3f %s\n", result.ID, result.CombinedScore, strings.TrimSpace(result.Content))
	}
	return nil
}

func docs() []document.Document {
	return []document.Document{
		{
			ID:     "go-gc",
			Title:  "go-garbage-collection.md",
			Format: document.FormatMarkdown,
			Text:   "Go uses a concurrent garbage collector to reclaim unreachable memory while keeping pauses short.",
		},
		{
			ID:     "pasta",
			Title:  "fresh-pasta.md",
			Format: document.FormatMarkdown,
			Text:   "Fresh pasta dough rests before rolling so tagliatelle holds sauce and keeps its bite.",
		},
		{
			ID:     "stars",
			Title:  "neutron-stars.md",
			Format: document.FormatMarkdown,
			Text:   "Neutron stars form from collapsed stellar cores and pack enormous mass into a tiny radius.",
		},
	}
}

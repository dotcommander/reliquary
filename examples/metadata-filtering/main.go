// Command metadata-filtering demonstrates Reliquary's structured metadata
// filtering during hybrid search.
//
// It shows:
// 1. Ingesting documents with structured metadata (tags, categories, environments).
// 2. Unfiltered hybrid retrieval.
// 3. Exact string and scalar filtering with WithFilter.
// 4. Multi-field conjunction filtering (AND semantics).
// 5. Reserved field filtering by filename and document ID.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/metadata-filtering
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	"github.com/dotcommander/reliquary/retrieval"
)

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	app := reliquary.Quickstart()

	docs := []document.Document{
		{
			ID:     "doc-auth-prod",
			Title:  "auth-service.md",
			Format: document.FormatMarkdown,
			Text:   "Authentication service handles OAuth2 tokens, refresh cycles, and session revocation.",
			Metadata: document.Metadata{
				"env":     "prod",
				"service": "auth",
			},
		},
		{
			ID:     "doc-auth-staging",
			Title:  "auth-staging.md",
			Format: document.FormatMarkdown,
			Text:   "Authentication staging deployment with mock token issuers and synthetic user accounts.",
			Metadata: document.Metadata{
				"env":     "staging",
				"service": "auth",
			},
		},
		{
			ID:     "doc-billing-prod",
			Title:  "billing-service.md",
			Format: document.FormatMarkdown,
			Text:   "Billing service manages stripe webhooks, invoice calculation, and payment reconciliation.",
			Metadata: document.Metadata{
				"env":     "prod",
				"service": "billing",
			},
		},
	}

	if _, err := app.Ingest(ctx, docs...); err != nil {
		return fmt.Errorf("ingest: %w", err)
	}

	query := "service tokens and payments"

	// 1. Unfiltered search: returns matches across all environments and services
	fmt.Printf("=== 1. Unfiltered Hybrid Search: %q ===\n", query)
	hits, err := app.Search(ctx, query, reliquary.TopK(3))
	if err != nil {
		return err
	}
	printHits(hits)

	// 2. Filter by environment scalar: env == "prod"
	fmt.Println("\n=== 2. Filter by env == \"prod\" ===")
	hits, err = app.Search(ctx, query,
		reliquary.WithFilter(map[string]any{"env": "prod"}),
		reliquary.TopK(3),
	)
	if err != nil {
		return err
	}
	printHits(hits)

	// 3. Multi-field conjunction: env == "prod" AND service == "auth"
	fmt.Println("\n=== 3. Conjunction Filter (env == \"prod\" AND service == \"auth\") ===")
	hits, err = app.Search(ctx, query,
		reliquary.WithFilter(map[string]any{
			"env":     "prod",
			"service": "auth",
		}),
		reliquary.TopK(3),
	)
	if err != nil {
		return err
	}
	printHits(hits)

	// 4. Reserved field filter: filename == "billing-service.md"
	fmt.Println("\n=== 4. Reserved Field Filter (filename == \"billing-service.md\") ===")
	hits, err = app.Search(ctx, query,
		reliquary.WithFilter(map[string]any{"filename": "billing-service.md"}),
		reliquary.TopK(3),
	)
	if err != nil {
		return err
	}
	printHits(hits)

	return nil
}

func printHits(hits []*retrieval.Result) {
	if len(hits) == 0 {
		fmt.Println("  (no results matching filter)")
		return
	}
	for i, hit := range hits {
		content := strings.ReplaceAll(strings.TrimSpace(hit.Content), "\n", " ")
		if len(content) > 70 {
			content = content[:70] + "..."
		}
		fmt.Printf("  [%d] %s (score: %.3f, env=%v): %s\n",
			i+1, hit.ID, hit.CombinedScore, hit.Metadata["env"], content)
	}
}

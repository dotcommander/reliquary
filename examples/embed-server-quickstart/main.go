// Command embed-server-quickstart wires a local OpenAI-compatible
// embedding server into the reliquary facade.
//
// Tier 0 of the Embed→Judge stack: a quantized local embedding model
// (default nomicai-modernbert-embed-base-4bit) served behind an
// OpenAI-compatible /v1/embeddings endpoint embeds the corpus and every
// query at near-zero marginal cost. Nomic-family models require the
// `search_document: ` and `search_query: ` task prefixes, applied by the
// adapter from the request kind; the server honors the `dimensions`
// field, so Matryoshka truncation (for example a second 64-d embedder for
// approximate candidate search) is a second constructor with a different
// Dimensions value.
//
// The caller owns the server process, endpoint, and credentials. Start
// your local embeddings server first (defaults below target
// http://localhost:8000/v1), then run from the repo root:
//
//	GOWORK=off go run ./examples/embed-server-quickstart
//
// Override with EMBED_SERVER_URL, EMBED_SERVER_API_KEY, EMBED_MODEL, and
// EMBED_DIM.
package main

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/adapter/openai"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	baseURL := getenv("EMBED_SERVER_URL", "http://localhost:8000/v1")
	apiKey := getenv("EMBED_SERVER_API_KEY", "local")
	model := getenv("EMBED_MODEL", "nomicai-modernbert-embed-base-4bit")
	dims := getenvInt("EMBED_DIM", 768)

	client := openaisdk.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	)
	embedder, err := openai.New(client, openai.Config{
		Model:          model,
		Dimensions:     dims,
		DocumentPrefix: "search_document: ",
		QueryPrefix:    "search_query: ",
	})
	if err != nil {
		return err
	}

	identity := fmt.Sprintf("embed-server-quickstart/%s@%d", model, dims)
	app, err := reliquary.New(
		reliquary.WithEmbedder(embedder),
		reliquary.WithIndexIdentity(identity),
	)
	if err != nil {
		return err
	}

	if _, err := app.Ingest(ctx, corpus()...); err != nil {
		return fmt.Errorf("ingest (is the embedding server running at %s?): %w", baseURL, err)
	}

	query := "what keeps go garbage collection pauses short"
	found, err := app.Search(ctx, query, reliquary.TopK(2))
	if err != nil {
		return err
	}

	fmt.Printf("Server: %s (%s, %d dims)\n", baseURL, model, dims)
	fmt.Printf("Query: %s\n", query)
	for _, result := range found {
		fmt.Printf("- [%.3f] %s\n", result.CombinedScore, result.Content)
	}
	return nil
}

func corpus() []document.Document {
	return []document.Document{
		{
			ID:     "go-gc",
			Title:  "go-garbage-collection.md",
			Format: document.FormatMarkdown,
			Text:   "Go uses a concurrent garbage collector to reclaim unreachable memory while keeping pauses short.",
		},
		{
			ID:     "nomic-prefix",
			Title:  "nomic-task-prefixes.md",
			Format: document.FormatMarkdown,
			Text:   "Nomic embedding models take search_document and search_query task prefixes so indexed text and queries land in comparable regions of the vector space.",
		},
		{
			ID:     "pasta",
			Title:  "fresh-pasta.md",
			Format: document.FormatMarkdown,
			Text:   "Fresh pasta dough rests before rolling so tagliatelle holds sauce and keeps its bite.",
		},
	}
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return fallback
}

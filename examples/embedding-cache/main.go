// Command embedding-cache demonstrates decorating an embedding.Embedder with
// Reliquary's transparent embedding cache (embedding/cache + inmem.Store).
//
// It shows:
//  1. Wrapping an embedder with an in-memory Store.
//  2. Monitoring base provider invocation counts.
//  3. Demonstrating that repeated document ingest and query calls hit the cache
//     with zero duplicate provider calls.
//
// Run from the repository root:
//
//	GOWORK=off go run ./examples/embedding-cache
package main

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embed"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/embedding/cache"
	"github.com/dotcommander/reliquary/embedding/cache/inmem"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
)

func main() {
	examplekit.Run(run)
}

// countedEmbedder wraps an embedder and counts actual provider embed calls.
type countedEmbedder struct {
	base  embedding.Embedder
	calls atomic.Int64
}

func (c *countedEmbedder) Embed(ctx context.Context, req embedding.Request) (embedding.Result, error) {
	c.calls.Add(1)
	return c.base.Embed(ctx, req)
}

func run(ctx context.Context) error {
	const dim = 64
	base := &countedEmbedder{base: embed.NewHashing(dim)}

	// 1. Construct the in-memory cache store and wrap the base embedder
	store := inmem.New()
	cachedEmbedder, err := cache.New(base, store, cache.Config{
		Identity: "demo-embedder-cache:v1",
	})
	if err != nil {
		return fmt.Errorf("new embedding cache: %w", err)
	}

	app, err := reliquary.New(
		reliquary.WithEmbedder(cachedEmbedder),
		reliquary.WithIndexIdentity("cache-demo-v1"),
	)
	if err != nil {
		return fmt.Errorf("new reliquary app: %w", err)
	}

	docs := []document.Document{
		{
			ID:     "doc-caching",
			Title:  "caching.md",
			Format: document.FormatMarkdown,
			Text:   "Embedding caches eliminate redundant external API calls for duplicate document chunks.",
		},
	}

	// 2. First Ingest: Cache Miss -> Base embedder called
	fmt.Println("=== 1. First Ingest (Cold Cache) ===")
	_, err = app.Ingest(ctx, docs...)
	if err != nil {
		return err
	}
	fmt.Printf("Base embedder call count after first ingest: %d\n", base.calls.Load())

	// 3. Second Ingest of identical text: Cache Hit -> Zero new base embedder calls
	fmt.Println("\n=== 2. Second Ingest (Warm Cache) ===")
	_, err = app.Ingest(ctx, docs...)
	if err != nil {
		return err
	}
	fmt.Printf("Base embedder call count after second ingest: %d (no change!)\n", base.calls.Load())

	// 4. Query Embedding Cache Check
	query := "redundant external API calls"
	fmt.Println("\n=== 3. First Search Query (Cold Query Cache) ===")
	callsBeforeSearch := base.calls.Load()
	_, err = app.Search(ctx, query, reliquary.TopK(1))
	if err != nil {
		return err
	}
	fmt.Printf("Calls made for first search query: %d\n", base.calls.Load()-callsBeforeSearch)

	fmt.Println("\n=== 4. Repeated Search Query (Warm Query Cache) ===")
	callsBeforeRepeat := base.calls.Load()
	_, err = app.Search(ctx, query, reliquary.TopK(1))
	if err != nil {
		return err
	}
	fmt.Printf("Calls made for repeated search query: %d (served from cache!)\n", base.calls.Load()-callsBeforeRepeat)

	fmt.Printf("\nTotal base provider embed calls across entire session: %d\n", base.calls.Load())
	return nil
}

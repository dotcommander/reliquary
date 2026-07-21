// Command rag-ingest-retrieve demonstrates an end-to-end retrieval pipeline
// assembled from reliquary packages the way a real consumer would:
//
//	document -> chunking -> embedding(Embedder seam) -> retrieval(hybrid + MMR)
//
// It ingests a tiny in-memory markdown corpus, chunks each document, embeds the
// chunks through a zero-dependency deterministic embedder (the feature-hashing
// trick), and answers a free-text query by hybrid scoring plus MMR
// diversification.
//
// The demo uses retrieval's small embedding helpers so the application code does
// not need to hand-convert vector widths or manually adapt MMR inputs.
//
// Run from the repo root:
//
//	GOWORK=off go run ./examples/rag-ingest-retrieve
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embed"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/examples/internal/examplekit"
	"github.com/dotcommander/reliquary/retrieval"
)

// embedDim is the width of the demo embedding space. Real models are far larger;
// 64 is enough to separate three small documents deterministically.
const embedDim = 64

func main() {
	examplekit.Run(run)
}

func run(ctx context.Context) error {
	// A miniature corpus mixing three unrelated topics so the ranking has signal.
	corpus := []document.Document{
		{
			ID:     "gc",
			Title:  "garbage-collection.md",
			Format: document.FormatMarkdown,
			Text: `# Garbage Collection in Go

The Go runtime uses a concurrent, tri-color mark-and-sweep garbage collector.
It tracks object reachability from root pointers and reclaims memory that is no
longer referenced by the program. The collector runs concurrently with user
goroutines to keep pause times low and predictable. A write barrier records
pointer mutations while marking proceeds in parallel across worker threads.`,
		},
		{
			ID:     "ragu",
			Title:  "ragu-recipe.md",
			Format: document.FormatMarkdown,
			Text: `# Ragù alla Bolognese

A proper ragù simmers ground meat with soffritto, tomatoes, and a splash of wine
until the sauce turns dense and silky. Basil and a pinch of nutmeg round out the
flavor. Serve over fresh tagliatelle, never spaghetti, so the sauce clings to the
ribbon-shaped pasta. Cook it slowly: the best ragù takes hours, not minutes.`,
		},
		{
			ID:     "neutron",
			Title:  "neutron-stars.md",
			Format: document.FormatMarkdown,
			Text: `# Neutron Stars

A neutron star forms when the core of a massive star collapses under gravity
during a supernova. Electrons and protons crush together into neutrons, packing
more than a solar mass into a sphere only twenty kilometers across. The collapse
leaves behind an object of extraordinary density with a powerful magnetic field.`,
		},
	}

	// 1. Chunk each document into retrieval.Result items. SmartBoundary treats
	//    the size argument as a max-character target while respecting sentence
	//    and word boundaries; overlap=0 keeps chunks disjoint.
	items, err := retrieval.ResultsFromDocuments(corpus, chunking.SmartBoundary, 220, 0)
	if err != nil {
		return err
	}
	fmt.Printf("Ingested %d document(s) into %d chunk(s)\n", len(corpus), len(items))

	// 2. Embed every chunk through the embedding.Embedder contract. This is the
	//    seam a real provider (ONNX, OpenAI, ...) plugs into; here we use a
	//    deterministic hashing-trick embedder so the demo has no dependencies.
	embedder := embed.NewHashing(embedDim)

	// 3. Attach provider-neutral embedding vectors to retrieval results.
	if err := retrieval.EmbedResults(ctx, embedder, items); err != nil {
		return err
	}

	// 4. Embed the query through the identical path.
	query := "how does the Go garbage collector reclaim memory"
	qRes, err := embedder.Embed(ctx, embedding.Request{Model: embedder.Model, Inputs: []string{query}})
	if err != nil {
		return err
	}

	// 5. Hybrid-score (embedding similarity + keyword overlap + filename boost),
	//    then diversify the top-k with Maximal Marginal Relevance.
	scorer := retrieval.NewScorer(retrieval.DefaultWeights())
	ranked := scorer.RerankEmbedding(qRes.Vectors[0], query, items)

	const topK = 3
	diversified := retrieval.Diversify(ranked, topK, 0.5)

	fmt.Printf("\nQuery: %q\n", query)
	fmt.Printf("Top-%d after MMR (lambda=0.5 = balanced relevance/diversity):\n", topK)
	for _, r := range diversified {
		fmt.Printf("\n  %s  combined=%.3f  (emb=%.3f kw=%.3f file=%.3f)\n",
			r.ID, r.CombinedScore, r.EmbeddingScore, r.KeywordScore, r.FilenameScore)
		fmt.Printf("  > %s\n", strings.Join(strings.Fields(strings.TrimSpace(r.Content)), " "))
	}
	return nil
}

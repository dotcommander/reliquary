package reliquary

import (
	"context"
	"fmt"
	"slices"

	"github.com/dotcommander/reliquary/chunking"
	"github.com/dotcommander/reliquary/chunkservice"
	"github.com/dotcommander/reliquary/document"
	"github.com/dotcommander/reliquary/embedding"
	"github.com/dotcommander/reliquary/retrieval"
)

// ChunkingOptions fixes the ingestion chunking policy for an App. ChunkSize
// uses strategy-specific units; MaxTokens independently caps final chunk text.
// The caller's index identity must describe the effective policy, including
// semantic options, encoding, ordered separators, and counter policy/version.
type ChunkingOptions struct {
	Strategy         chunking.Strategy
	ChunkSize        int
	Overlap          int
	Semantic         *chunkservice.SemanticOptions
	TokenEncoding    chunkservice.TokenEncoding
	MaxTokens        int
	SeparatorProfile chunkservice.SeparatorProfile
	CustomSeparators []string
	// PreflightCounter remains caller-owned. Its behavior must stay stable
	// throughout the App's lifetime and support the caller's concurrency pattern.
	// It is separate from the tokenizer-backed hard ceiling and billing usage.
	PreflightCounter chunking.TokenCounter
}

// WithChunking selects fixed chunkservice orchestration. It cannot be combined
// with WithChunker. New validates the policy without embedding or index access.
// Semantic options and separator slices are copied when the option is applied.
func WithChunking(options ChunkingOptions) Option {
	return func(a *App) {
		policy := options
		policy.CustomSeparators = slices.Clone(options.CustomSeparators)
		if options.Semantic != nil {
			semantic := *options.Semantic
			policy.Semantic = &semantic
		}
		a.chunkingPolicy = &policy
	}
}

func (o ChunkingOptions) request(text string) chunkservice.Request {
	return chunkservice.Request{
		Text: text, Strategy: o.Strategy, ChunkSize: o.ChunkSize, Overlap: o.Overlap,
		Semantic: o.Semantic, TokenEncoding: o.TokenEncoding, MaxTokens: o.MaxTokens,
		SeparatorProfile: o.SeparatorProfile, CustomSeparators: o.CustomSeparators,
		PreflightCounter: o.PreflightCounter,
	}
}

func (a *App) configureChunking() error {
	if a.chunkingPolicy == nil {
		return nil
	}
	if a.chunkerSet {
		return fmt.Errorf("reliquary: WithChunker and WithChunking cannot be combined")
	}
	if err := a.chunkingPolicy.request("").ValidatePolicy(); err != nil {
		return fmt.Errorf("reliquary: chunking policy: %w", err)
	}
	// EmbedResults uses the embedder's default model and document input kind.
	bridge := chunking.BatchEmbedderFromEmbedder(a.embedder, embedding.ModelRef{}, embedding.KindDocument)
	a.chunkService = chunkservice.New().WithEmbedder(bridge)
	return nil
}

func (a *App) chunkDocuments(ctx context.Context, docs []document.Document) ([]*retrieval.Result, error) {
	if a.chunkingPolicy == nil {
		return retrieval.ResultsFromDocuments(docs, a.strategy, a.size, a.overlap)
	}
	var results []*retrieval.Result
	for _, doc := range docs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		normalized := document.NormalizeText(doc.Text)
		if normalized == "" {
			continue
		}
		chunks, _, err := a.chunkService.Chunks(ctx, a.chunkingPolicy.request(normalized))
		if err != nil {
			return nil, fmt.Errorf("reliquary: ingest document %q: chunking: %w", doc.ID, err)
		}
		results = append(results, retrieval.ResultsFromChunks(doc, normalized, chunks)...)
	}
	return results, nil
}

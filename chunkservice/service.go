package chunkservice

import (
	"context"
	"fmt"
	"log/slog"
	"unicode/utf8"

	"github.com/dotcommander/reliquary/chunking"
)

// Service chunks text. The optional embedder enables the semantic strategy
// and include_embeddings payloads; without it, both fail validation clearly.
type Service struct {
	embedder       chunking.BatchEmbedder
	embeddingModel string
	log            *slog.Logger
}

// New returns a service without an embedder: the semantic strategy and
// include_embeddings fail validation with a clear error.
func New() *Service {
	return &Service{log: discardLogger()}
}

// WithLogger routes service logs (requests that fail mid-processing, batch
// item failures) to l. Nil restores the silent default. Chainable.
func (s *Service) WithLogger(l *slog.Logger) *Service {
	if l == nil {
		l = discardLogger()
	}
	s.log = l
	return s
}

// WithEmbedder backs the semantic strategy and include_embeddings payloads
// with the given embedder. Chainable.
func (s *Service) WithEmbedder(embedder chunking.BatchEmbedder) *Service {
	s.embedder = embedder
	return s
}

// WithEmbeddingModel records the provider model name reported in metadata
// when responses embed chunk text. Chainable for wiring convenience.
func (s *Service) WithEmbeddingModel(model string) *Service {
	s.embeddingModel = model
	return s
}

// Process validates req, runs it through the chunking engine, and projects
// the result onto the wire DTOs. The semantic strategy is strict: a mid-run
// embedding failure fails the request instead of silently degrading to
// boundary chunks, because a result labeled semantic must actually be
// semantic.
func (s *Service) Process(ctx context.Context, req Request) (*Response, error) {
	if req.IncludeEmbeddings && s.embedder == nil {
		return nil, EmbeddingsUnavailable()
	}
	libChunks, metadata, err := s.Chunks(ctx, req)
	if err != nil {
		return nil, err
	}
	chunks := FromLibChunks(req.Text, libChunks)
	if req.IncludeEmbeddings {
		if err := s.embedChunks(ctx, chunks, &metadata); err != nil {
			return nil, err
		}
	}
	return &Response{
		Chunks:   chunks,
		Metadata: metadata,
	}, nil
}

// Chunks runs req through the engine and returns the library chunks (byte
// spans, Path/Metadata/ContentHash annotations intact) plus response
// metadata, before wire projection. File-oriented callers use it to consume
// spans and annotations directly. Validation, semantic strictness, the
// legacy preflight budget, and the post-strategy token ceiling all apply.
func (s *Service) Chunks(ctx context.Context, req Request) ([]chunking.Chunk, Metadata, error) {
	if err := ctx.Err(); err != nil {
		return nil, Metadata{}, err
	}

	if err := req.Validate(); err != nil {
		return nil, Metadata{}, err
	}

	strategy := withDefaultStrategy(req.Strategy)

	var libChunks []chunking.Chunk
	var err error
	if strategy == chunking.Semantic {
		libChunks, err = s.semanticChunks(ctx, req)
	} else {
		libChunks, err = s.boundaryChunks(req, strategy)
	}
	if err != nil {
		return nil, Metadata{}, err
	}
	if req.MaxTokens > 0 {
		libChunks, err = chunking.EnforceTokenCeiling(libChunks, string(req.TokenEncoding.WithDefault()), req.MaxTokens)
		if err != nil {
			return nil, Metadata{}, fmt.Errorf("failed to enforce max_tokens: %w", err)
		}
	}

	totalChars := utf8.RuneCountInString(req.Text)
	totalTokens, err := calculateTotalTokens(req, strategy)
	if err != nil {
		return nil, Metadata{}, fmt.Errorf("count tokens: %w", err)
	}

	metadata := Metadata{
		TotalChunks:      len(libChunks),
		TotalChars:       totalChars,
		TotalTokens:      totalTokens,
		StrategyUsed:     strategy,
		TokenEncoding:    effectiveEncoding(req),
		MaxTokens:        req.MaxTokens,
		SeparatorProfile: req.SeparatorProfile,
	}
	return libChunks, metadata, nil
}

// semanticChunks runs the semantic strategy strictly. The strategy needs an
// embedding provider — without one, an explicit request fails clearly
// rather than falling back. A fresh latch per request: the engine's
// SemanticChunker swallows embedding errors and falls back to boundary
// chunking; the latch lets us refuse to emit fallback chunks as semantic.
// The optional preflight counter then splits any chunk that exceeds the
// caller's token budget.
func (s *Service) semanticChunks(ctx context.Context, req Request) ([]chunking.Chunk, error) {
	if s.embedder == nil {
		return nil, SemanticUnavailable()
	}
	latch := chunking.LatchBatchEmbedder(s.embedder)
	semanticChunker, err := chunking.NewSemanticChunker(latch, SemanticOptsFrom(req.Semantic))
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic chunker: %w", err)
	}
	chunks := semanticChunker.ChunkSemantic(ctx, req.Text, req.ChunkSize, req.Overlap)
	if err := latch.Err(); err != nil {
		return nil, fmt.Errorf("semantic strategy: embedding failed mid-run; refusing to emit fallback chunks as semantic: %w", err)
	}
	if req.PreflightCounter == nil {
		return chunks, nil
	}
	return chunking.EnforceTokenLimits(chunks, req.PreflightCounter), nil
}

// boundaryChunks runs every non-semantic strategy, weaving the optional
// preflight token budget into the chunking call itself.
func (s *Service) boundaryChunks(req Request, strategy chunking.Strategy) ([]chunking.Chunk, error) {
	chunker, err := buildChunker(strategy, req)
	if err != nil {
		return nil, err
	}
	if req.PreflightCounter == nil {
		return chunker.Chunk(req.Text, req.ChunkSize, req.Overlap), nil
	}
	return chunking.ChunkWithTokenLimit(chunker, req.Text, req.ChunkSize, req.Overlap, req.PreflightCounter), nil
}

// buildChunker constructs the chunker for one non-semantic strategy.
func buildChunker(strategy chunking.Strategy, req Request) (chunking.Chunker, error) {
	switch {
	case strategy == chunking.TokenBased:
		return chunking.NewTokenChunker(string(req.TokenEncoding.WithDefault()))
	case strategy == chunking.Recursive && req.CustomSeparators != nil:
		separators := append([]string(nil), req.CustomSeparators...)
		if separators[len(separators)-1] != "" {
			separators = append(separators, "")
		}
		return chunking.NewRecursiveChunker(chunking.SeparatorProfile{Separators: separators}), nil
	case strategy == chunking.Recursive && req.SeparatorProfile == ProfileCJKThai:
		return chunking.NewRecursiveChunker(chunking.CJKThaiSeparatorProfile()), nil
	default:
		chunker, err := chunking.NewChunker(strategy)
		if err != nil {
			return nil, fmt.Errorf("failed to create chunker: %w", err)
		}
		return chunker, nil
	}
}

// embedChunks embeds every chunk's final text in one batch call and attaches
// the vectors in order. Called after token-limit composition so the embedded
// text is exactly what the response carries. An explicit include_embeddings
// request fails on embedder error — no silent fallback.
func (s *Service) embedChunks(ctx context.Context, chunks []Chunk, metadata *Metadata) error {
	if len(chunks) == 0 {
		return nil
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	vectors, err := s.embedder.EmbedBatch(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed chunks: %w", err)
	}
	if len(vectors) != len(chunks) {
		return fmt.Errorf("embed chunks: provider returned %d vectors for %d chunks", len(vectors), len(chunks))
	}
	for i, v := range vectors {
		chunks[i].Embedding = v
	}
	metadata.EmbeddingModel = s.embeddingModel
	metadata.EmbeddingDimensions = len(vectors[0])
	return nil
}

// withDefaultStrategy resolves the effective strategy: an unset or unknown
// value becomes smart_boundary.
func withDefaultStrategy(s chunking.Strategy) chunking.Strategy {
	if s == "" || !ValidStrategy(s) {
		return chunking.SmartBoundary
	}
	return s
}

// effectiveEncoding reports the encoding metadata should echo: the
// request's when set, the default when a ceiling forced one, empty when
// token counting never engaged.
func effectiveEncoding(req Request) TokenEncoding {
	if req.TokenEncoding != "" {
		return req.TokenEncoding
	}
	if req.MaxTokens > 0 || req.Strategy == chunking.TokenBased {
		return req.TokenEncoding.WithDefault()
	}
	return ""
}

// calculateTotalTokens returns the token count of the ORIGINAL request text
// for token_based chunking, on the same basis as TotalChars (original text,
// not a sum across chunks). Summing per-chunk counts would double-count the
// overlap region; counting the source text once is consistent with TotalChars.
func calculateTotalTokens(req Request, strategy chunking.Strategy) (int, error) {
	if strategy != chunking.TokenBased {
		return 0, nil
	}
	encoding := string(req.TokenEncoding.WithDefault())
	return chunking.CountTokens(req.Text, encoding)
}

// FromLibChunk projects a library chunk into the wire DTO, preserving the
// strategy-produced annotations: Path (heading breadcrumb), Metadata
// (markdown block annotations), and ContentHash (16-hex SHA-256 of Text).
// start_char/end_char are converted from the library's authoritative byte
// span into rune offsets against source; unmappable spans normalize to (0,0).
func FromLibChunk(source string, c chunking.Chunk) Chunk {
	start, end := chunking.RuneSpan(source, c.StartChar, c.EndChar)
	return Chunk{
		ID:          c.ID,
		StartChar:   start,
		EndChar:     end,
		Text:        c.Text,
		CharCount:   c.CharCount,
		WordCount:   c.WordCount,
		TokenCount:  c.TokenCount,
		Path:        c.Path,
		Metadata:    c.Metadata,
		ContentHash: c.ContentHash,
	}
}

// FromLibChunks maps a slice of library chunks to wire DTOs against source.
func FromLibChunks(source string, in []chunking.Chunk) []Chunk {
	out := make([]Chunk, len(in))
	for i := range in {
		out[i] = FromLibChunk(source, in[i])
	}
	return out
}

// SemanticOptsFrom maps wire semantic options to the library type. Zero
// fields stay zero: the library applies its own defaults for unset values.
func SemanticOptsFrom(o *SemanticOptions) chunking.SemanticOpts {
	if o == nil {
		return chunking.SemanticOpts{}
	}
	return chunking.SemanticOpts{
		BreakSensitivity: o.BreakSensitivity,
		MinChunkChars:    o.MinChunkChars,
		MaxChunkChars:    o.MaxChunkChars,
		SmoothingWindow:  o.SmoothingWindow,
		CoherenceWindow:  o.CoherenceWindow,
	}
}

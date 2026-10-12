// Package chunkservice owns the request-level chunking orchestration over
// the chunking engine: request validation, strategy dispatch (including a
// strict semantic path), legacy preflight token budgets, post-strategy token
// ceilings, per-chunk embedding attachment, batch envelopes, and the wire
// projection of results. Applications (CLIs, HTTP servers) wire providers
// and presentation; this package decides how a request becomes chunks.
package chunkservice

import (
	"errors"
	"fmt"

	"github.com/dotcommander/reliquary/chunking"
)

type requestValidationError struct {
	message string
}

func (e *requestValidationError) Error() string {
	return e.message
}

func invalidRequest(format string, args ...any) error {
	return &requestValidationError{message: fmt.Sprintf(format, args...)}
}

// IsRequestValidationError reports whether err came from Request.Validate.
// Callers use this boundary to distinguish safe client errors from internal
// failures without coupling transport status codes to error-message text.
func IsRequestValidationError(err error) bool {
	var validationErr *requestValidationError
	return errors.As(err, &validationErr)
}

// Request is one chunking job: a text plus the parameters that shape it.
// Zero-value optionals mean "use the engine default". It backs both wire
// APIs and programmatic callers.
type Request struct {
	Text      string            `json:"text"`
	ChunkSize int               `json:"chunk_size"`
	Strategy  chunking.Strategy `json:"strategy,omitempty"`
	Overlap   int               `json:"overlap,omitempty"`
	// TokenEncoding names the tiktoken encoding used for token_based
	// chunking and the MaxTokens ceiling (default cl100k_base).
	TokenEncoding TokenEncoding `json:"token_encoding,omitempty"`
	// SeparatorProfile selects the recursive strategy's separator cascade:
	// "default_text" (empty) or "cjk_thai". Only valid with strategy=recursive.
	SeparatorProfile SeparatorProfile `json:"separator_profile,omitempty"`
	// Semantic tunes the semantic strategy; zero fields fall back to library
	// defaults. Only valid with strategy=semantic.
	Semantic *SemanticOptions `json:"semantic,omitempty"`
	// MaxTokens caps every chunk's token count (measured with TokenEncoding,
	// default cl100k_base) after the base strategy runs. 0 disables the cap.
	// Chunks that exceed it are split further (sentence, then word, then
	// hard cut), so strategies can stay semantic while honoring LLM context
	// limits. It is independent of chunk_size: chunk_size bounds characters
	// (tokens for token_based); max_tokens bounds tokens on every strategy.
	MaxTokens int `json:"max_tokens,omitempty"`
	// IncludeEmbeddings attaches an embedding vector to every chunk via the
	// configured embedding provider. Works with any strategy; requires
	// embedding configuration on the service (a clear validation error
	// otherwise). Unlike the semantic strategy, a mid-request embedding
	// failure fails the request rather than falling back: the caller
	// explicitly asked for vectors.
	IncludeEmbeddings bool `json:"include_embeddings,omitempty"`
	// PreflightCounter applies a caller-supplied token budget while chunking
	// (the legacy "preflight" pipeline): boundary strategies chunk with
	// token awareness and semantic output is split where the counter says a
	// chunk exceeds budget. Unlike MaxTokens it needs no tiktoken encoding;
	// callers bring their own counter. nil disables it.
	PreflightCounter chunking.TokenCounter `json:"-"`
}

// Validate enforces request invariants that the engine cannot express:
// required fields, range bounds, and strategy-coupled options.
func (r Request) Validate() error {
	if r.Text == "" {
		return invalidRequest("text is required")
	}

	if r.ChunkSize <= 0 {
		return invalidRequest("chunk_size must be greater than 0")
	}

	if r.Overlap < 0 {
		return invalidRequest("overlap must be greater than or equal to 0")
	}

	if r.Overlap >= r.ChunkSize {
		return invalidRequest("overlap must be less than chunk_size")
	}

	if r.Strategy != "" && !ValidStrategy(r.Strategy) {
		return invalidRequest("unknown strategy: %q", r.Strategy)
	}

	if r.TokenEncoding != "" && !r.TokenEncoding.IsValid() {
		return invalidRequest("unknown token_encoding: %q", r.TokenEncoding)
	}

	if r.MaxTokens < 0 {
		return invalidRequest("max_tokens must be greater than or equal to 0")
	}
	return r.validateStrategyCouplings()
}

// validateStrategyCouplings checks the options that are only meaningful
// with one specific strategy: separator_profile (recursive) and semantic
// (semantic).
func (r Request) validateStrategyCouplings() error {
	if r.SeparatorProfile != "" {
		if !r.SeparatorProfile.IsValid() {
			return invalidRequest("unknown separator_profile: %q", r.SeparatorProfile)
		}
		if r.Strategy != chunking.Recursive {
			return invalidRequest("separator_profile requires the recursive strategy")
		}
	}

	if r.Semantic != nil {
		if r.Strategy != chunking.Semantic {
			return invalidRequest("semantic options require the semantic strategy")
		}
		if err := r.Semantic.validate(); err != nil {
			return err
		}
	}

	return nil
}

// Response is the chunking result: the ordered chunks plus run metadata.
type Response struct {
	Chunks   []Chunk  `json:"chunks"`
	Metadata Metadata `json:"metadata"`
}

// Chunk is one chunked span of the request text. StartChar/EndChar are
// rune offsets into Text of the request.
type Chunk struct {
	ID         int    `json:"id"`
	StartChar  int    `json:"start_char"`
	EndChar    int    `json:"end_char"`
	Text       string `json:"text"`
	CharCount  int    `json:"char_count"`
	WordCount  int    `json:"word_count"`
	TokenCount int    `json:"token_count,omitempty"`
	// Path is the heading breadcrumb ("H1 Title" → "Subsection") for
	// heading-aware strategies; omitted otherwise.
	Path []string `json:"path,omitempty"`
	// Metadata carries strategy-produced annotations (for example
	// markdown_aware's heading_level, language, line_count); omitted when the
	// strategy produces none.
	Metadata map[string]string `json:"metadata,omitempty"`
	// ContentHash is the first 16 hex characters of SHA-256(Text).
	ContentHash string `json:"content_hash"`
	// Embedding is the chunk text's embedding vector, present only when the
	// request set include_embeddings.
	Embedding []float32 `json:"embedding,omitempty"`
}

// Metadata summarizes one chunking run for callers: totals, the strategy
// actually applied, and the option echoes that changed behavior.
type Metadata struct {
	TotalChunks   int               `json:"total_chunks"`
	TotalChars    int               `json:"total_chars"`
	TotalTokens   int               `json:"total_tokens,omitempty"`
	StrategyUsed  chunking.Strategy `json:"strategy_used"`
	TokenEncoding TokenEncoding     `json:"token_encoding,omitempty"`
	// MaxTokens reports the token ceiling that was enforced, set only when
	// the request asked for one.
	MaxTokens int `json:"max_tokens,omitempty"`
	// SeparatorProfile reports the recursive separator cascade in effect,
	// set only when the request specified one.
	SeparatorProfile SeparatorProfile `json:"separator_profile,omitempty"`
	// EmbeddingModel names the provider model that produced the chunk
	// embeddings; set only when the request set include_embeddings.
	EmbeddingModel string `json:"embedding_model,omitempty"`
	// EmbeddingDimensions is the vector length of the chunk embeddings;
	// set only when the request set include_embeddings.
	EmbeddingDimensions int `json:"embedding_dimensions,omitempty"`
}

// ValidStrategy reports whether s names a strategy the service dispatches.
// The zero value is valid (it means "use the default").
func ValidStrategy(s chunking.Strategy) bool {
	switch s {
	case chunking.WordBoundary, chunking.SentenceBoundary, chunking.HardCut,
		chunking.ParagraphAware, chunking.TokenBased, chunking.SmartBoundary,
		chunking.MarkdownAware, chunking.HeadingAware, chunking.Recursive,
		chunking.Semantic:
		return true
	default:
		return false
	}
}

// SemanticUnavailable returns the request-validation error reported when the
// semantic strategy is requested while no embedding configuration exists.
// The semantic strategy needs an embedder; without one, an explicit request
// must fail clearly, not silently fall back.
func SemanticUnavailable() error {
	return invalidRequest("semantic strategy requires embedding configuration")
}

// EmbeddingsUnavailable returns the request-validation error reported when
// include_embeddings is requested while no embedding configuration exists.
// The flag needs a provider; failing clearly beats silently omitting vectors.
func EmbeddingsUnavailable() error {
	return invalidRequest("include_embeddings requires embedding configuration")
}

// TokenEncoding names a tiktoken encoding for token counting.
type TokenEncoding string

// The supported tiktoken encodings.
const (
	EncodingCL100K TokenEncoding = "cl100k_base" // GPT-3.5/GPT-4
	EncodingO200K  TokenEncoding = "o200k_base"  // gpt-5
	EncodingP50K   TokenEncoding = "p50k_base"   // Older models
	EncodingR50K   TokenEncoding = "r50k_base"   // Very old models
)

// IsValid reports whether e names a supported encoding.
func (e TokenEncoding) IsValid() bool {
	switch e {
	case EncodingCL100K, EncodingO200K, EncodingP50K, EncodingR50K:
		return true
	default:
		return false
	}
}

// WithDefault resolves the effective encoding: unset or unknown values
// become cl100k_base.
func (e TokenEncoding) WithDefault() TokenEncoding {
	if e == "" || !e.IsValid() {
		return EncodingCL100K
	}
	return e
}

// SeparatorProfile selects the recursive strategy separator cascade by its
// wire identifier; the profiles themselves live in the chunking package.
type SeparatorProfile string

// The recursive separator cascade identifiers.
const (
	ProfileDefaultText SeparatorProfile = chunking.SeparatorProfileDefaultTextID
	ProfileCJKThai     SeparatorProfile = chunking.SeparatorProfileCJKThaiID
)

// IsValid reports whether p names a supported cascade.
func (p SeparatorProfile) IsValid() bool {
	switch p {
	case ProfileDefaultText, ProfileCJKThai:
		return true
	default:
		return false
	}
}

// SemanticOptions tunes the semantic strategy. Zero values mean library
// defaults (see chunking.SemanticOpts): MaxChunkChars 1600, MinChunkChars
// 200, BreakSensitivity 1.0; smoothing and coherence windows are disabled
// by default.
type SemanticOptions struct {
	BreakSensitivity float64 `json:"break_sensitivity,omitempty"`
	MinChunkChars    int     `json:"min_chunk_chars,omitempty"`
	MaxChunkChars    int     `json:"max_chunk_chars,omitempty"`
	SmoothingWindow  int     `json:"smoothing_window,omitempty"`
	CoherenceWindow  int     `json:"coherence_window,omitempty"`
}

func (o *SemanticOptions) validate() error {
	if o.BreakSensitivity < 0 {
		return invalidRequest("semantic.break_sensitivity must be greater than or equal to 0")
	}
	if o.MinChunkChars < 0 || o.MaxChunkChars < 0 {
		return invalidRequest("semantic chunk char bounds must be greater than or equal to 0")
	}
	if o.MinChunkChars > 0 && o.MaxChunkChars > 0 && o.MaxChunkChars < o.MinChunkChars {
		return invalidRequest("semantic.max_chunk_chars must be greater than or equal to semantic.min_chunk_chars")
	}
	if o.SmoothingWindow < 0 || o.CoherenceWindow < 0 {
		return invalidRequest("semantic windows must be greater than or equal to 0")
	}
	return nil
}

// BatchRequest chunks many texts under one shared parameter set. Each text
// is processed independently: item-level failures (for example an empty
// text) surface as per-item error results, while batch-level problems (bad
// shared params, missing texts) fail the whole request.
type BatchRequest struct {
	Texts             []string          `json:"texts"`
	ChunkSize         int               `json:"chunk_size"`
	Strategy          chunking.Strategy `json:"strategy,omitempty"`
	Overlap           int               `json:"overlap,omitempty"`
	TokenEncoding     TokenEncoding     `json:"token_encoding,omitempty"`
	SeparatorProfile  SeparatorProfile  `json:"separator_profile,omitempty"`
	Semantic          *SemanticOptions  `json:"semantic,omitempty"`
	MaxTokens         int               `json:"max_tokens,omitempty"`
	IncludeEmbeddings bool              `json:"include_embeddings,omitempty"`
}

// ToChunkRequest converts the shared parameters plus one text into a single
// chunk request — the exact transformation batch processing applies per item.
func (r BatchRequest) ToChunkRequest(text string) Request {
	return Request{
		Text:              text,
		ChunkSize:         r.ChunkSize,
		Strategy:          r.Strategy,
		Overlap:           r.Overlap,
		TokenEncoding:     r.TokenEncoding,
		SeparatorProfile:  r.SeparatorProfile,
		Semantic:          r.Semantic,
		MaxTokens:         r.MaxTokens,
		IncludeEmbeddings: r.IncludeEmbeddings,
	}
}

// Validate checks the batch shape and the shared parameters once. The probe
// text is non-empty so only shared fields are exercised; per-text validity
// (empty text) is an item-level result, not a whole-batch failure.
func (r BatchRequest) Validate() error {
	if len(r.Texts) == 0 {
		return invalidRequest("texts must contain at least one text")
	}
	return r.ToChunkRequest("probe text").Validate()
}

// BatchResult carries exactly one of Response or Error for one input text,
// keyed by its index in the request — no partial-failure ambiguity.
type BatchResult struct {
	Index    int       `json:"index"`
	Response *Response `json:"response,omitempty"`
	Error    string    `json:"error,omitempty"`
}

// BatchResponse is the batch result: one entry per input text, in request
// order.
type BatchResponse struct {
	Results []BatchResult `json:"results"`
}

package chunking

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"unicode/utf8"
)

const (
	contentHashAlgo = "sha256"
	contentHashLen  = 8 // bytes before hex encoding (16 hex chars)
)

// Chunk represents a segment of text produced by a chunking strategy.
type Chunk struct {
	ID         int
	Text       string
	CharCount  int
	WordCount  int
	TokenCount int
	// StartChar and EndChar are byte offsets into the original input text,
	// such that originalText[StartChar:EndChar] == Text when the chunk text
	// appears verbatim in the source. For overlap chunks, the span points to
	// the chunk's non-overlap contribution. Spans are rebased from
	// section-relative to original-relative coordinates by adjustChunkSpans
	// (base.go). Spans are cleared (set to 0) when a post-processing step
	// (e.g. EnforceHardLimits) cannot map the result back to the original text.
	StartChar int
	EndChar   int
	// Path is the section breadcrumb from headings; nil for non-heading-aware strategies.
	Path []string
	// Metadata holds block-type metadata from goldmark parsing; nil for non-goldmark strategies.
	Metadata map[string]string
	// ContentHash is the first 16 hex characters of SHA-256(text); always set.
	ContentHash string
}

// Strategy identifies a chunking algorithm.
type Strategy string

const (
	SmartBoundary    Strategy = "smart_boundary"
	SentenceBoundary Strategy = "sentence_boundary"
	WordBoundary     Strategy = "word_boundary"
	MarkdownAware    Strategy = "markdown_aware"
	HeadingAware     Strategy = "heading_aware"
	ParagraphAware   Strategy = "paragraph_aware"
	HardCut          Strategy = "hard_cut"
	TokenBased       Strategy = "token_based"
	Semantic         Strategy = "semantic"
)

// ErrUnknownStrategy is returned by NewChunker when an unrecognized strategy
// name is provided.
var ErrUnknownStrategy = errors.New("chunking: unknown strategy")

// Chunker splits text into segments according to a strategy.
type Chunker interface {
	Chunk(text string, size int, overlap int) []Chunk
	Strategy() Strategy
}

// NewChunker creates a Chunker for the given strategy.
func NewChunker(strategy Strategy) (Chunker, error) {
	switch strategy {
	case SmartBoundary:
		return newSmartBoundaryChunker(), nil
	case SentenceBoundary:
		return newSentenceBoundaryChunker(), nil
	case WordBoundary:
		return newWordBoundaryChunker(), nil
	case MarkdownAware:
		return newMarkdownAwareChunker(), nil
	case HeadingAware:
		return newHeadingAwareChunker(), nil
	case ParagraphAware:
		return newParagraphAwareChunker(), nil
	case HardCut:
		return newHardCutChunker(), nil
	case Optimal:
		return NewOptimalChunker(), nil
	case TokenBased:
		return newTokenBasedChunker(), nil
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnknownStrategy, strategy)
	}
}

// NewTokenChunker creates a Chunker that splits text at token boundaries using
// the specified tiktoken encoding. Empty encoding defaults to "cl100k_base".
// Returns an error if the encoding name is not recognized by tiktoken.
func NewTokenChunker(encoding string) (Chunker, error) {
	if encoding == "" {
		encoding = defaultEncoding
	}
	// Validate the encoding by attempting to load it.
	if _, err := getEncoder(encoding); err != nil {
		return nil, fmt.Errorf("invalid token encoding %q: %w", encoding, err)
	}
	return &tokenBasedChunker{encoding: encoding}, nil
}

// buildChunk constructs a Chunk with computed counts, content hash, and no source span.
func buildChunk(id int, text string) Chunk {
	hash := sha256.Sum256([]byte(text))
	return Chunk{
		ID:          id,
		Text:        text,
		CharCount:   utf8.RuneCountInString(text),
		WordCount:   countWords(text),
		ContentHash: fmt.Sprintf("%x", hash[:contentHashLen]),
	}
}

// buildChunkWithSpan constructs a Chunk with computed counts, content hash, and source byte offsets.
// start and end must be byte offsets into the original input text.
func buildChunkWithSpan(id int, text string, start, end int) Chunk {
	hash := sha256.Sum256([]byte(text))
	return Chunk{
		ID:          id,
		Text:        text,
		CharCount:   utf8.RuneCountInString(text),
		WordCount:   countWords(text),
		StartChar:   start,
		EndChar:     end,
		ContentHash: fmt.Sprintf("%x", hash[:contentHashLen]),
	}
}

package chunking

import (
	"fmt"
	"strings"
	"sync"

	"github.com/pkoukk/tiktoken-go"
)

// Default token encoding for LLM chunking.
const defaultEncoding = "cl100k_base"

// encoderCache caches tiktoken encoders by name. GetEncoding is expensive
// (~26ms) because it builds a regex token encoder from BPE vocabulary on each call.
var encoderCache sync.Map // map[string]*tiktoken.Tiktoken

// Tokenizer is the consumer-owned tokenization boundary used for chunk sizing
// and preflight estimates. Implementations must use the tokenizer associated
// with the target provider or local model.
type Tokenizer interface {
	Encode(text string) ([]int, error)
	Count(text string) (int, error)
}

// TiktokenTokenizer provides OpenAI-compatible preflight estimates. API
// responses remain authoritative for actual request usage.
type TiktokenTokenizer struct {
	encoding string
}

// NewTiktokenTokenizer creates an OpenAI preflight tokenizer for encoding.
// Empty encoding defaults to cl100k_base.
func NewTiktokenTokenizer(encoding string) (*TiktokenTokenizer, error) {
	if encoding == "" {
		encoding = defaultEncoding
	}
	if _, err := getEncoder(encoding); err != nil {
		return nil, fmt.Errorf("invalid token encoding %q: %w", encoding, err)
	}
	return &TiktokenTokenizer{encoding: encoding}, nil
}

// Encode returns token IDs for text.
func (t *TiktokenTokenizer) Encode(text string) ([]int, error) {
	if t == nil {
		return nil, fmt.Errorf("tiktoken tokenizer is nil")
	}
	enc, err := getEncoder(t.encoding)
	if err != nil {
		return nil, fmt.Errorf("get tiktoken encoding %q: %w", t.encoding, err)
	}
	return enc.Encode(text, nil, nil), nil
}

// Count returns the number of tokens in text.
func (t *TiktokenTokenizer) Count(text string) (int, error) {
	tokens, err := t.Encode(text)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}

// decode is intentionally private: decoding is needed by the concrete token
// chunker, but is not part of the replaceable counting contract.
func (t *TiktokenTokenizer) decode(tokens []int) (string, error) {
	if t == nil {
		return "", fmt.Errorf("tiktoken tokenizer is nil")
	}
	enc, err := getEncoder(t.encoding)
	if err != nil {
		return "", fmt.Errorf("get tiktoken encoding %q: %w", t.encoding, err)
	}
	return enc.Decode(tokens), nil
}

// getEncoder returns a cached tiktoken encoder, creating it on first access.
// LoadOrStore ensures only one goroutine's value wins under concurrent calls;
// any duplicate encoder is discarded and the stored value is returned.
func getEncoder(encoding string) (*tiktoken.Tiktoken, error) {
	if v, ok := encoderCache.Load(encoding); ok {
		return v.(*tiktoken.Tiktoken), nil
	}
	enc, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, err
	}
	actual, _ := encoderCache.LoadOrStore(encoding, enc)
	return actual.(*tiktoken.Tiktoken), nil
}

// tokenBasedChunker splits text at token boundaries using tiktoken.
type tokenBasedChunker struct {
	encoding string
}

func newTokenBasedChunker() *tokenBasedChunker {
	return &tokenBasedChunker{encoding: defaultEncoding}
}

func (t *tokenBasedChunker) Strategy() Strategy {
	return TokenBased
}

// Chunk splits text into chunks of at most `size` tokens with `overlap` token overlap.
// Falls back to word-boundary splitting if the encoding is unavailable.
func (t *tokenBasedChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}

	tokenizer, err := NewTiktokenTokenizer(t.encoding)
	if err != nil {
		// Approximate 4 chars per token and fall back to word-boundary splitting.
		return t.fallback(text, size*4, overlap*4)
	}

	tokens, err := tokenizer.Encode(text)
	if err != nil {
		return t.fallback(text, size*4, overlap*4)
	}
	if len(tokens) == 0 {
		return nil
	}

	var chunks []Chunk
	chunkID := 0
	cursor := 0

	for i := 0; i < len(tokens); {
		start := i

		// Apply overlap from the previous chunk.
		if chunkID > 0 && overlap > 0 {
			start = i - overlap
			if start < 0 {
				start = 0
			}
		}

		end := i + size
		if end > len(tokens) {
			end = len(tokens)
		}

		chunkTokens := tokens[start:end]
		chunkText, err := tokenizer.decode(chunkTokens)
		if err != nil {
			return t.fallback(text, size*4, overlap*4)
		}

		// Clean up boundaries for mid-text splits.
		if end < len(tokens) {
			chunkText = cleanChunkEnd(chunkText)
		}
		if start > 0 {
			chunkText = cleanChunkStart(chunkText)
		}

		if chunkText != "" {
			startChar, endChar := findTokenChunkSpan(text, chunkText, cursor)
			if endChar > cursor {
				cursor = endChar
			}
			chunk := buildChunkWithSpan(chunkID, chunkText, startChar, endChar)
			chunk.TokenCount = len(chunkTokens)
			chunks = append(chunks, chunk)
			chunkID++
		}

		i = end
	}

	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size * 6, Overlap: overlap, OriginalText: text})
}

// findTokenChunkSpan locates byte offsets for chunkText within original starting
// from cursor. Uses monotonic forward search so repeated token sequences get
// correct, non-overlapping spans. Returns (0, 0) if no match is found.
func findTokenChunkSpan(original, chunkText string, cursor int) (int, int) {
	if cursor < 0 {
		cursor = 0
	}
	idx := strings.Index(original[cursor:], chunkText)
	if idx >= 0 {
		s := cursor + idx
		return s, s + len(chunkText)
	}
	return 0, 0
}

func (t *tokenBasedChunker) fallback(text string, size int, overlap int) []Chunk {
	chunker := newWordBoundaryChunker()
	return chunker.Chunk(text, size, overlap)
}

// cleanChunkStart removes a likely partial word at the start of a chunk.
func cleanChunkStart(text string) string {
	idx := strings.IndexFunc(text, func(r rune) bool {
		return r == ' ' || r == '\n' || r == '\t'
	})
	if idx > 0 && idx < 10 {
		return text[idx+1:]
	}
	return text
}

// cleanChunkEnd tries to end the chunk at a sentence or word boundary.
func cleanChunkEnd(text string) string {
	lastPeriod := strings.LastIndexAny(text, ".!?")
	lastSpace := strings.LastIndexAny(text, " \n\t")

	if lastPeriod > 0 && lastPeriod > len(text)-20 {
		return text[:lastPeriod+1]
	}
	if lastSpace > 0 && lastSpace > len(text)-10 {
		return text[:lastSpace]
	}
	return text
}

// CountTokens returns the token count for text using the named tiktoken encoding.
// The encoder is cached after first use for the given encoding name.
func CountTokens(text string, encoding string) (int, error) {
	tokenizer, err := NewTiktokenTokenizer(encoding)
	if err != nil {
		return 0, err
	}
	return tokenizer.Count(text)
}

// FillTokenCounts sets TokenCount on each chunk whose TokenCount is 0.
// Uses the named tiktoken encoding (e.g. "cl100k_base"). Chunks that
// already have a non-zero TokenCount are skipped.
func FillTokenCounts(chunks []Chunk, encoding string) error {
	tokenizer, err := NewTiktokenTokenizer(encoding)
	if err != nil {
		return err
	}
	return FillTokenCountsWithTokenizer(chunks, tokenizer)
}

// FillTokenCountsWithTokenizer fills missing counts using the caller-selected
// provider or model tokenizer.
func FillTokenCountsWithTokenizer(chunks []Chunk, tokenizer Tokenizer) error {
	if tokenizer == nil {
		return fmt.Errorf("tokenizer is nil")
	}
	for i := range chunks {
		if chunks[i].TokenCount == 0 && chunks[i].Text != "" {
			count, err := tokenizer.Count(chunks[i].Text)
			if err != nil {
				return fmt.Errorf("count chunk %d tokens: %w", chunks[i].ID, err)
			}
			chunks[i].TokenCount = count
		}
	}
	return nil
}

// TiktokenCounter implements TokenCounter using a cached tiktoken encoder.
// Create one with NewTiktokenCounter.
type TiktokenCounter struct {
	tokenizer *TiktokenTokenizer
	maxTokens int
}

// NewTiktokenCounter creates a TiktokenCounter for the given encoding and
// maximum token budget. Empty encoding defaults to "cl100k_base".
// maxTokens <= 0 disables token limiting (pass-through in EnforceTokenLimits).
// Returns an error if the encoding name is not recognized by tiktoken.
func NewTiktokenCounter(encoding string, maxTokens int) (*TiktokenCounter, error) {
	tokenizer, err := NewTiktokenTokenizer(encoding)
	if err != nil {
		return nil, err
	}
	return &TiktokenCounter{tokenizer: tokenizer, maxTokens: maxTokens}, nil
}

// CountTokens returns the number of tokens in text using the configured encoding.
// Returns 0 if the counter is nil or if the encoder fails (should be unreachable
// after constructor validation).
func (t *TiktokenCounter) CountTokens(text string) int {
	if t == nil || t.tokenizer == nil {
		return 0
	}
	count, err := t.tokenizer.Count(text)
	if err != nil {
		return 0
	}
	return count
}

// MaxTokens returns the maximum allowed tokens per chunk.
// Returns 0 if the counter is nil, which disables token limiting.
func (t *TiktokenCounter) MaxTokens() int {
	if t == nil {
		return 0
	}
	return t.maxTokens
}

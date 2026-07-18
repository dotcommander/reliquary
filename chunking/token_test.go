package chunking

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCountTokens_Basic(t *testing.T) {
	t.Parallel()

	n, err := CountTokens("Hello, world!", "cl100k_base")
	if err != nil {
		t.Fatalf("CountTokens error: %v", err)
	}
	// "Hello, world!" is 4 tokens in cl100k_base: "Hello", ",", " world", "!"
	if n != 4 {
		t.Errorf("CountTokens(\"Hello, world!\") = %d, want 4", n)
	}
}

func TestCountTokens_Empty(t *testing.T) {
	t.Parallel()

	n, err := CountTokens("", "cl100k_base")
	if err != nil {
		t.Fatalf("CountTokens error: %v", err)
	}
	if n != 0 {
		t.Errorf("CountTokens(\"\") = %d, want 0", n)
	}
}

func TestCountTokens_Estimate(t *testing.T) {
	t.Parallel()

	// A known paragraph -- verify the token count is reasonable.
	// cl100k_base averages ~4 chars/token for English text.
	text := "The quick brown fox jumps over the lazy dog. " +
		"This sentence has several words and should produce a predictable token count."

	n, err := CountTokens(text, "cl100k_base")
	if err != nil {
		t.Fatalf("CountTokens error: %v", err)
	}

	charCount := len([]rune(text))
	// Sanity check: tokens should be between charCount/6 and charCount/2.
	minTokens := charCount / 6
	maxTokens := charCount / 2

	if n < minTokens || n > maxTokens {
		t.Errorf("CountTokens = %d for %d chars, expected between %d and %d",
			n, charCount, minTokens, maxTokens)
	}
}

func TestCountTokens_InvalidEncoding(t *testing.T) {
	t.Parallel()

	_, err := CountTokens("test", "nonexistent_encoding")
	if err == nil {
		t.Fatal("expected error for invalid encoding, got nil")
	}
}

func TestNewTokenChunker_DefaultEncoding(t *testing.T) {
	t.Parallel()

	c, err := NewTokenChunker("")
	if err != nil {
		t.Fatalf("NewTokenChunker(\"\") error: %v", err)
	}
	if c.Strategy() != TokenBased {
		t.Errorf("Strategy() = %q, want %q", c.Strategy(), TokenBased)
	}
	// Produces same chunks as NewChunker(TokenBased).
	text := "Hello world, this is a test of token chunking."
	chunks := c.Chunk(text, 10, 0)
	for _, ch := range chunks {
		if ch.TokenCount <= 0 {
			t.Errorf("chunk %d: TokenCount = %d, want > 0", ch.ID, ch.TokenCount)
		}
	}
}

func TestNewTokenChunker_O200K(t *testing.T) {
	t.Parallel()

	c, err := NewTokenChunker("o200k_base")
	if err != nil {
		t.Fatalf("NewTokenChunker(\"o200k_base\") error: %v", err)
	}

	text := "The quick brown fox jumps over the lazy dog."
	chunks := c.Chunk(text, 8, 0)
	for _, ch := range chunks {
		if ch.TokenCount <= 0 {
			t.Errorf("chunk %d: TokenCount = %d, want > 0", ch.ID, ch.TokenCount)
		}
	}
}

func TestNewTokenChunker_P50K(t *testing.T) {
	t.Parallel()

	c, err := NewTokenChunker("p50k_base")
	if err != nil {
		t.Fatalf("NewTokenChunker(\"p50k_base\") error: %v", err)
	}

	text := "A simple test string for tokenization."
	chunks := c.Chunk(text, 10, 0)
	for _, ch := range chunks {
		if ch.TokenCount <= 0 {
			t.Errorf("chunk %d: TokenCount = %d, want > 0", ch.ID, ch.TokenCount)
		}
	}
}

func TestNewTokenChunker_R50K(t *testing.T) {
	t.Parallel()

	c, err := NewTokenChunker("r50k_base")
	if err != nil {
		t.Fatalf("NewTokenChunker(\"r50k_base\") error: %v", err)
	}

	text := "Another test for the r50k token encoder."
	chunks := c.Chunk(text, 10, 0)
	for _, ch := range chunks {
		if ch.TokenCount <= 0 {
			t.Errorf("chunk %d: TokenCount = %d, want > 0", ch.ID, ch.TokenCount)
		}
	}
}

func TestNewTokenChunker_InvalidEncoding(t *testing.T) {
	t.Parallel()

	c, err := NewTokenChunker("not-real")
	if err == nil {
		t.Fatal("expected error for invalid encoding, got nil")
	}
	if c != nil {
		t.Error("expected nil chunker for invalid encoding")
	}
}

func TestNewChunker_TokenBased_StillDefault(t *testing.T) {
	t.Parallel()

	// Verify NewChunker(TokenBased) still uses cl100k_base.
	c, err := NewChunker(TokenBased)
	if err != nil {
		t.Fatalf("NewChunker(TokenBased) error: %v", err)
	}

	text := "Hello world, this is a test of token chunking."
	chunks := c.Chunk(text, 10, 0)
	if len(chunks) == 0 {
		t.Fatal("expected non-empty chunks")
	}
	for _, ch := range chunks {
		if ch.TokenCount <= 0 {
			t.Errorf("chunk %d: TokenCount = %d, want > 0", ch.ID, ch.TokenCount)
		}
	}
}

func TestFillTokenCounts_FillsZero(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 0, Text: "Hello, world!", TokenCount: 0},
		{ID: 1, Text: "Another sentence.", TokenCount: 0},
	}

	err := FillTokenCounts(chunks, "cl100k_base")
	if err != nil {
		t.Fatalf("FillTokenCounts error: %v", err)
	}

	for _, c := range chunks {
		if c.TokenCount <= 0 {
			t.Errorf("chunk %d: TokenCount = %d, want > 0", c.ID, c.TokenCount)
		}
	}
}

func TestFillTokenCounts_SkipsNonZero(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 0, Text: "Hello!", TokenCount: 42},
		{ID: 1, Text: "World.", TokenCount: 0},
	}

	err := FillTokenCounts(chunks, "cl100k_base")
	if err != nil {
		t.Fatalf("FillTokenCounts error: %v", err)
	}

	if chunks[0].TokenCount != 42 {
		t.Errorf("chunk 0: TokenCount = %d, want 42 (should not overwrite)", chunks[0].TokenCount)
	}
	if chunks[1].TokenCount <= 0 {
		t.Errorf("chunk 1: TokenCount = %d, want > 0", chunks[1].TokenCount)
	}
}

func TestFillTokenCounts_NilSlice(t *testing.T) {
	t.Parallel()

	err := FillTokenCounts(nil, "cl100k_base")
	if err != nil {
		t.Fatalf("FillTokenCounts(nil) error: %v", err)
	}
}

func TestFillTokenCounts_BadEncoding(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{{ID: 0, Text: "test", TokenCount: 0}}
	err := FillTokenCounts(chunks, "nonexistent_encoding")
	if err == nil {
		t.Fatal("expected error for bad encoding, got nil")
	}
}

func TestTokenChunkerDuplicateSpans(t *testing.T) {
	t.Parallel()

	// Text with a repeated phrase — old bare strings.Index would pin both
	// chunks to the first occurrence. The cursor-tracked approach must give
	// each chunk its own span.
	text := "foo bar foo bar foo bar foo bar foo bar foo bar"

	c, err := NewTokenChunker("cl100k_base")
	if err != nil {
		t.Fatalf("NewTokenChunker error: %v", err)
	}

	// Use small token size to force multiple chunks with repeated content.
	chunks := c.Chunk(text, 5, 0)
	if len(chunks) < 2 {
		t.Fatalf("expected >= 2 chunks, got %d", len(chunks))
	}

	// Verify monotonic spans: each chunk's StartChar >= previous EndChar.
	for i := 1; i < len(chunks); i++ {
		if chunks[i].StartChar > 0 && chunks[i-1].EndChar > 0 {
			if chunks[i].StartChar < chunks[i-1].EndChar {
				t.Errorf("chunk %d StartChar=%d < chunk %d EndChar=%d (spans not monotonic)",
					i, chunks[i].StartChar, i-1, chunks[i-1].EndChar)
			}
		}
	}

	// Verify round-trip: text[StartChar:EndChar] == chunk.Text for non-zero spans.
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue // zero span = unknown
		}
		if ch.EndChar > len(text) {
			t.Errorf("chunk %d: EndChar=%d exceeds text len=%d", ch.ID, ch.EndChar, len(text))
			continue
		}
		got := text[ch.StartChar:ch.EndChar]
		if got != ch.Text {
			t.Errorf("chunk %d: text[%d:%d] = %q, want %q",
				ch.ID, ch.StartChar, ch.EndChar, got, ch.Text)
		}
	}
}

func TestNewTiktokenCounter_DefaultEncoding(t *testing.T) {
	t.Parallel()

	tc, err := NewTiktokenCounter("", 10)
	require.NoError(t, err)
	require.NotNil(t, tc)

	assert.Equal(t, 10, tc.MaxTokens())
	assert.Greater(t, tc.CountTokens("hello world"), 0)
}

func TestNewTiktokenCounter_InvalidEncoding(t *testing.T) {
	t.Parallel()

	tc, err := NewTiktokenCounter("not_real_encoding", 10)
	assert.Error(t, err)
	assert.Nil(t, tc)
}

func TestTiktokenCounter_DisabledMaxTokens(t *testing.T) {
	t.Parallel()

	tc, err := NewTiktokenCounter("", 0)
	require.NoError(t, err)

	assert.Equal(t, 0, tc.MaxTokens())

	// maxTokens=0 means disabled — EnforceTokenLimits should pass through.
	chunks := []Chunk{{ID: 0, Text: "some text here", TokenCount: 0}}
	result := EnforceTokenLimits(chunks, tc)
	assert.Len(t, result, 1)
}

func TestTiktokenCounter_NilSafe(t *testing.T) {
	t.Parallel()

	var tc *TiktokenCounter
	assert.Equal(t, 0, tc.MaxTokens())
	assert.Equal(t, 0, tc.CountTokens("anything"))
}

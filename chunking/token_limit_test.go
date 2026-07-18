package chunking

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeTokenCounter counts tokens as whitespace-delimited words for testing.
type fakeTokenCounter struct {
	maxTokens int
}

func (f fakeTokenCounter) CountTokens(text string) int {
	return len(strings.Fields(text))
}

func (f fakeTokenCounter) MaxTokens() int {
	return f.maxTokens
}

type byteTokenCounter struct {
	maxTokens int
}

func (b byteTokenCounter) CountTokens(text string) int {
	return len(text)
}

func (b byteTokenCounter) MaxTokens() int {
	return b.maxTokens
}

func TestEnforceTokenLimits_AllFitFastPath(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		buildChunk(0, "short text"),
		buildChunk(1, "another chunk"),
	}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 100})

	require.Len(t, result, 2)
	assert.Equal(t, "short text", result[0].Text)
	assert.Equal(t, "another chunk", result[1].Text)
}

func TestEnforceTokenLimits_ShortTextFastPath(t *testing.T) {
	t.Parallel()

	// Text shorter than maxTokens*2 chars should skip tokenization entirely.
	chunks := []Chunk{
		buildChunk(0, "hi"),
	}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 100})

	require.Len(t, result, 1)
	assert.Equal(t, "hi", result[0].Text)
}

func TestEnforceTokenLimits_OversizedSentenceSplit(t *testing.T) {
	t.Parallel()

	// Each word = 1 token. MaxTokens = 3.
	text := "alpha beta gamma delta. epsilon zeta eta theta."
	chunks := []Chunk{buildChunk(0, text)}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 3})

	// Every resulting chunk must fit within 3 tokens.
	for _, c := range result {
		tokenCount := len(strings.Fields(c.Text))
		assert.LessOrEqual(t, tokenCount, 3,
			"chunk %d exceeds token limit: %d tokens, text %q", c.ID, tokenCount, c.Text)
	}

	// No empty chunks.
	for _, c := range result {
		assert.NotEmpty(t, strings.TrimSpace(c.Text))
	}
}

func TestEnforceTokenLimits_OversizedWordSplit(t *testing.T) {
	t.Parallel()

	// No sentence boundaries — all words.
	text := "alpha beta gamma delta epsilon zeta"
	chunks := []Chunk{buildChunk(0, text)}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 2})

	for _, c := range result {
		tokenCount := len(strings.Fields(c.Text))
		assert.LessOrEqual(t, tokenCount, 2,
			"chunk %d exceeds token limit: %d tokens", c.ID, tokenCount)
	}
}

func TestEnforceTokenLimits_IndivisibleToken(t *testing.T) {
	t.Parallel()

	// A single word longer than maxTokens — must still appear.
	longWord := strings.Repeat("x", 300)
	chunks := []Chunk{buildChunk(0, longWord)}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 1})

	assert.NotEmpty(t, result, "indivisible token should still produce output")
	assert.Equal(t, longWord, result[0].Text)
}

func TestEnforceTokenLimits_OversizedSingleAtomHardCut(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{buildChunk(0, "abcdefghij")}

	result := EnforceTokenLimits(chunks, byteTokenCounter{maxTokens: 3})

	require.NotEmpty(t, result)
	assert.Equal(t, []string{"abc", "def", "ghi", "j"}, chunkTexts(result))
	for _, c := range result {
		assert.LessOrEqual(t, len(c.Text), 3,
			"chunk %d exceeds byte token limit: %q", c.ID, c.Text)
		assert.Equal(t, 0, c.StartChar)
		assert.Equal(t, 0, c.EndChar)
	}
}

func TestEnforceTokenLimits_EmptyChunksDropped(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		buildChunk(0, ""),
		buildChunk(1, "actual content"),
		buildChunk(2, "   "),
	}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 100})

	require.Len(t, result, 1)
	assert.Equal(t, "actual content", result[0].Text)
}

func TestEnforceTokenLimits_IDsRenumbered(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 5, Text: "aaa bbb"},
		{ID: 10, Text: "ccc ddd"},
	}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 100})

	require.Len(t, result, 2)
	assert.Equal(t, 0, result[0].ID)
	assert.Equal(t, 1, result[1].ID)
}

func TestEnforceTokenLimits_ZeroMaxTokensPassthrough(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{buildChunk(0, "some text")}
	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 0})

	require.Len(t, result, 1)
	assert.Equal(t, "some text", result[0].Text)
}

func TestEnforceTokenLimits_SpansClearedOnSplit(t *testing.T) {
	t.Parallel()

	// Chunk with span that exceeds token limit.
	text := "alpha beta gamma delta epsilon"
	chunks := []Chunk{
		{ID: 0, Text: text, StartChar: 5, EndChar: 35, CharCount: 30},
	}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 2})

	// Sub-chunks from split should have cleared spans.
	for _, c := range result {
		assert.Equal(t, 0, c.StartChar, "split sub-chunk %d should have StartChar=0", c.ID)
		assert.Equal(t, 0, c.EndChar, "split sub-chunk %d should have EndChar=0", c.ID)
	}
}

func TestEnforceTokenLimits_SpansPreservedOnPassThrough(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 0, Text: "short", StartChar: 10, EndChar: 15, CharCount: 5},
	}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 100})

	require.Len(t, result, 1)
	assert.Equal(t, 10, result[0].StartChar)
	assert.Equal(t, 15, result[0].EndChar)
}

func TestEnforceTokenLimits_TextOrderPreserved(t *testing.T) {
	t.Parallel()

	text := "alpha beta gamma delta epsilon zeta eta theta"
	chunks := []Chunk{buildChunk(0, text)}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 2})

	// Concatenating all chunk texts should reproduce the original words.
	var allWords []string
	for _, c := range result {
		allWords = append(allWords, strings.Fields(c.Text)...)
	}

	expected := strings.Fields(text)
	assert.Equal(t, expected, allWords)
}

// ---------------------------------------------------------------------------
// ID sequencing after token-limit splits
// ---------------------------------------------------------------------------

func TestEnforceTokenLimits_IDsSequentialAfterSplit(t *testing.T) {
	t.Parallel()

	// Single chunk that requires splitting — verify IDs are sequential.
	text := "alpha beta gamma delta epsilon zeta eta theta"
	chunks := []Chunk{buildChunk(0, text)}

	result := EnforceTokenLimits(chunks, fakeTokenCounter{maxTokens: 2})

	require.NotEmpty(t, result)

	for i, c := range result {
		assert.Equal(t, i, c.ID,
			"chunk at index %d has ID=%d, want %d (sequential from 0)", i, c.ID, i)
	}

	// Verify no duplicate IDs.
	seen := make(map[int]bool)
	for _, c := range result {
		assert.False(t, seen[c.ID], "duplicate ID %d", c.ID)
		seen[c.ID] = true
	}
}

func chunkTexts(chunks []Chunk) []string {
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	return texts
}

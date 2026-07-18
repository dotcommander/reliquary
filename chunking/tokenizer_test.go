package chunking

import (
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubTokenizer struct {
	tokens []int
	err    error
}

type nonlinearTokenizer struct{}

type failAfterTokenizer struct {
	remaining int
	err       error
}

func (t *failAfterTokenizer) Encode(text string) ([]int, error) {
	count, err := t.Count(text)
	return make([]int, count), err
}

func (t *failAfterTokenizer) Count(string) (int, error) {
	if t.remaining == 0 {
		return 0, t.err
	}
	t.remaining--
	return 1, nil
}

func (nonlinearTokenizer) Encode(text string) ([]int, error) {
	count, err := nonlinearTokenizer{}.Count(text)
	return make([]int, count), err
}

func (nonlinearTokenizer) Count(text string) (int, error) {
	words := len(strings.Fields(text))
	return words * words, nil
}

func (s stubTokenizer) Encode(string) ([]int, error) {
	if s.err != nil {
		return nil, s.err
	}
	return append([]int(nil), s.tokens...), nil
}

func (s stubTokenizer) Count(text string) (int, error) {
	tokens, err := s.Encode(text)
	return len(tokens), err
}

func TestTiktokenTokenizerImplementsTokenizer(t *testing.T) {
	t.Parallel()

	tokenizer, err := NewTiktokenTokenizer("")
	require.NoError(t, err)

	var _ Tokenizer = tokenizer
	tokens, err := tokenizer.Encode("Hello, world!")
	require.NoError(t, err)
	count, err := tokenizer.Count("Hello, world!")
	require.NoError(t, err)
	assert.NotEmpty(t, tokens)
	assert.Equal(t, len(tokens), count)
}

func TestFillTokenCountsWithTokenizerUsesCallerImplementation(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{{ID: 7, Text: "provider-specific text"}}
	err := FillTokenCountsWithTokenizer(chunks, stubTokenizer{tokens: []int{11, 12, 13}})

	require.NoError(t, err)
	assert.Equal(t, 3, chunks[0].TokenCount)
}

func TestFillTokenCountsWithTokenizerPropagatesError(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("model tokenizer failed")
	chunks := []Chunk{{ID: 7, Text: "provider-specific text"}}
	err := FillTokenCountsWithTokenizer(chunks, stubTokenizer{err: wantErr})

	require.Error(t, err)
	assert.ErrorIs(t, err, wantErr)
	assert.Contains(t, err.Error(), "chunk 7")
	assert.Zero(t, chunks[0].TokenCount)
}

func TestFillTokenCountsWithTokenizerRejectsNil(t *testing.T) {
	t.Parallel()

	err := FillTokenCountsWithTokenizer(nil, nil)
	require.Error(t, err)
}

func TestEnforceTokenLimitsWithTokenizerUsesCallerImplementation(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{buildChunk(0, "short text")}
	result, err := EnforceTokenLimitsWithTokenizer(chunks, stubTokenizer{tokens: []int{1, 2, 3}}, 3)

	require.NoError(t, err)
	require.Len(t, result, 1)
	assert.Equal(t, "short text", result[0].Text)
}

func TestEnforceTokenLimitsWithTokenizerPropagatesError(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("model tokenizer failed")
	chunks := []Chunk{buildChunk(0, "short text")}
	result, err := EnforceTokenLimitsWithTokenizer(chunks, stubTokenizer{err: wantErr}, 3)

	assert.Nil(t, result)
	assert.ErrorIs(t, err, wantErr)
}

func TestEnforceTokenLimitsWithTokenizerRechecksNonlinearCounts(t *testing.T) {
	t.Parallel()

	tokenizer := nonlinearTokenizer{}
	chunks := []Chunk{buildChunk(0, "one two three four")}
	result, err := EnforceTokenLimitsWithTokenizer(chunks, tokenizer, 5)

	require.NoError(t, err)
	require.NotEmpty(t, result)
	for _, chunk := range result {
		count, countErr := tokenizer.Count(chunk.Text)
		require.NoError(t, countErr)
		assert.LessOrEqual(t, count, 5, "chunk %q exceeds exact budget", chunk.Text)
	}
}

func TestEnforceTokenLimitsWithTokenizerPropagatesVerificationError(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("verification failed")
	tokenizer := &failAfterTokenizer{remaining: 1, err: wantErr}
	result, err := EnforceTokenLimitsWithTokenizer([]Chunk{buildChunk(0, "text")}, tokenizer, 5)

	assert.Nil(t, result)
	assert.ErrorIs(t, err, wantErr)
}

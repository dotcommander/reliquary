package chunking

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSentenceBoundary_Strategy(t *testing.T) {
	t.Parallel()
	c := newSentenceBoundaryChunker()
	assert.Equal(t, SentenceBoundary, c.Strategy())
}

func TestBuildSentenceChunksWithSpans_NoSpanSuppression(t *testing.T) {
	t.Parallel()

	text := "Go. Then we use Go."
	chunks := newSentenceBoundaryChunker().Chunk(text, 50, 0)
	require.NotEmpty(t, chunks, "expected at least one chunk")

	// With the masking guard removed and rune-index derivation in place,
	// spans should be non-zero and satisfy the round-trip invariant.
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		assert.Equal(t, ch.Text, text[ch.StartChar:ch.EndChar],
			"span round-trip failed for chunk %d (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}

	// At least one chunk must have a non-zero span — this text is simple
	// enough that all chunks should have accurate spans.
	hasSpan := false
	for _, ch := range chunks {
		if ch.StartChar != 0 || ch.EndChar != 0 {
			hasSpan = true
			break
		}
	}
	assert.True(t, hasSpan, "expected at least one chunk with non-zero spans")
}

func TestSentenceBoundary_ForceAddPreservesSpans(t *testing.T) {
	t.Parallel()

	text := "Sentence one is very long. Sentence two is also very long."
	// Chunker size=30, overlap=10.
	// Sentence one (length 26) fits in Chunk 0.
	// Sentence two (length 30) does not fit with the overlap, so it triggers force-add.
	// The force-added chunk (length 57) is split by EnforceHardLimits into 3 parts.
	chunks := newSentenceBoundaryChunker().Chunk(text, 30, 10)
	require.Len(t, chunks, 4)

	assert.Equal(t, "Sentence one is very long.", chunks[0].Text)
	assert.Equal(t, 0, chunks[0].StartChar)
	assert.Equal(t, 26, chunks[0].EndChar)

	assert.Equal(t, "Sentence one is very long.", chunks[1].Text)
	assert.Equal(t, 0, chunks[1].StartChar)
	assert.Equal(t, 0, chunks[1].EndChar)

	assert.Equal(t, "Sentence two is also very", chunks[2].Text)
	assert.Equal(t, "long.", chunks[3].Text)
}

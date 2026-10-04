package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRecursive_Strategy(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	assert.Equal(t, Recursive, c.Strategy())

	registered, err := NewChunker(Recursive)
	require.NoError(t, err)
	assert.Equal(t, Recursive, registered.Strategy())
}

func TestRecursive_DefaultProfileForEmptyConstructor(t *testing.T) {
	t.Parallel()
	text := "First paragraph.\n\nSecond paragraph. Also words here.\n\nThird."
	fromDefault := NewRecursiveChunker(DefaultTextSeparatorProfile()).Chunk(text, 30, 0)
	fromEmpty := NewRecursiveChunker(SeparatorProfile{ID: "empty"}).Chunk(text, 30, 0)

	require.NotEmpty(t, fromEmpty)
	require.Len(t, fromEmpty, len(fromDefault))
	for i := range fromEmpty {
		assert.Equal(t, fromDefault[i].Text, fromEmpty[i].Text)
	}
}

func TestRecursive_EmptyAndInvalidInput(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	assert.Nil(t, c.Chunk("", 100, 0))
	assert.Nil(t, c.Chunk("hello", 0, 0))
	assert.Nil(t, c.Chunk("hello", -3, 0))
}

func TestRecursive_SmallTextSingleChunk(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	chunks := c.Chunk("Short text.", 100, 0)
	require.Len(t, chunks, 1)
	assert.Equal(t, "Short text.", chunks[0].Text)
	assert.Equal(t, 0, chunks[0].StartChar)
	assert.Equal(t, len("Short text."), chunks[0].EndChar)
}

func TestRecursive_ParagraphCascade(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	text := "First paragraph here.\n\nSecond paragraph here."
	chunks := c.Chunk(text, 24, 0)

	require.Len(t, chunks, 2)
	assert.Equal(t, "First paragraph here.", chunks[0].Text)
	assert.Equal(t, "Second paragraph here.", chunks[1].Text)
	for _, ch := range chunks {
		require.True(t, ch.EndChar > ch.StartChar)
		assert.Equal(t, ch.Text, text[ch.StartChar:ch.EndChar])
	}
}

func TestRecursive_WordCascade(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	chunks := c.Chunk("alpha beta gamma delta epsilon zeta", 12, 0)

	require.Len(t, chunks, 4)
	assert.Equal(t, "alpha beta", chunks[0].Text)
	assert.Equal(t, "gamma delta", chunks[1].Text)
	assert.Equal(t, "epsilon", chunks[2].Text)
	assert.Equal(t, "zeta", chunks[3].Text)
}

func TestRecursive_OverlapSharesTailRunes(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	const overlap = 4
	chunks := c.Chunk("alpha beta gamma delta epsilon zeta", 12, overlap)

	require.Greater(t, len(chunks), 2)
	for i := 1; i < len(chunks); i++ {
		prevRunes := []rune(chunks[i-1].Text)
		curRunes := []rune(chunks[i].Text)
		require.GreaterOrEqual(t, len(prevRunes), overlap)
		require.GreaterOrEqual(t, len(curRunes), overlap)
		assert.Equal(t,
			string(prevRunes[len(prevRunes)-overlap:]),
			string(curRunes[:overlap]),
			"chunk %d must start with the previous chunk's %d-rune tail", i, overlap)
	}
	for _, ch := range chunks {
		assert.LessOrEqual(t, utf8.RuneCountInString(ch.Text), 12)
	}
}

func TestRecursive_HardCutFallback(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	chunks := c.Chunk(strings.Repeat("a", 50), 10, 0)

	require.Len(t, chunks, 5)
	for _, ch := range chunks {
		assert.Equal(t, 10, ch.CharCount)
	}
}

func TestRecursive_OversizedPieceWithoutHardCutSeparator(t *testing.T) {
	t.Parallel()
	profile := SeparatorProfile{ID: "no_hard_cut", Separators: []string{"\n\n", "\n", " "}}
	c := NewRecursiveChunker(profile)
	chunks := c.Chunk(strings.Repeat("a", 50), 10, 0)

	// The unbreakable leaf is emitted whole, then EnforceHardLimits cascades
	// to a hard cut, so every output chunk still fits the budget.
	total := 0
	require.NotEmpty(t, chunks)
	for _, ch := range chunks {
		assert.LessOrEqual(t, ch.CharCount, 10)
		total += ch.CharCount
	}
	assert.Equal(t, 50, total)
}

func TestRecursive_CJKThaiProfile(t *testing.T) {
	t.Parallel()
	c := NewRecursiveChunker(CJKThaiSeparatorProfile())
	text := "first。second。third。"
	chunks := c.Chunk(text, 8, 0)

	require.GreaterOrEqual(t, len(chunks), 2)
	joined := strings.Join(chunkTexts(chunks), "|")
	assert.Contains(t, joined, "first")
	assert.Contains(t, joined, "second")
	assert.Contains(t, joined, "third")
	for _, ch := range chunks {
		if ch.EndChar > ch.StartChar {
			assert.Equal(t, ch.Text, text[ch.StartChar:ch.EndChar])
		}
	}
}

func TestRecursive_UTF8BoundarySafety(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	text := strings.Repeat("😀", 12)
	for _, overlap := range []int{0, 1} {
		chunks := c.Chunk(text, 4, overlap)
		require.NotEmpty(t, chunks, "overlap=%d", overlap)
		for _, ch := range chunks {
			assert.True(t, utf8.ValidString(ch.Text), "chunk text must be valid UTF-8")
			if ch.EndChar > ch.StartChar {
				assert.Equal(t, ch.Text, text[ch.StartChar:ch.EndChar])
			}
		}
	}
}

func TestRecursive_AllChunksFitBudget(t *testing.T) {
	t.Parallel()
	c := newRecursiveChunker()
	tests := []struct {
		name string
		text string
		size int
		olap int
	}{
		{"paragraphs", "one two three\n\nfour five six\n\nseven eight nine", 12, 0},
		{"words", "w1 w2 w3 w4 w5 w6 w7 w8 w9", 8, 2},
		{"mixed", "# Heading\n\nBody text here.\n\nMore body text.", 18, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			chunks := c.Chunk(tt.text, tt.size, tt.olap)
			require.NotEmpty(t, chunks)
			for _, ch := range chunks {
				assert.LessOrEqual(t, ch.CharCount, tt.size,
					"chunk %d exceeds budget: %q", ch.ID, ch.Text)
				if ch.EndChar > ch.StartChar {
					assert.Equal(t, ch.Text, tt.text[ch.StartChar:ch.EndChar])
				}
			}
		})
	}
}

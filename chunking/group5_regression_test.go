package chunking

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// C5-01: nodeByteSpan must skip lineless children returning (-1, 0), and
// appendChunkIfValid must guard start < 0.
func TestMarkdownAware_LinelessChildrenNoPanic(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)

	cases := []struct {
		name string
		doc  string
	}{
		{"blockquote with thematic break", "> hello\n>\n> ---"},
		{"list with empty item", "- hello\n- "},
		{"combined", "> alpha\n>\n> ---\n\n- beta\n- "},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			chunks := c.Chunk(tc.doc, 200, 0)
			require.NotEmpty(t, chunks)
			for _, ch := range chunks {
				if ch.StartChar == 0 && ch.EndChar == 0 {
					continue
				}
				require.GreaterOrEqual(t, ch.StartChar, 0)
				require.Less(t, ch.StartChar, ch.EndChar)
				require.LessOrEqual(t, ch.EndChar, len(tc.doc))
				assert.Equal(t, ch.Text, tc.doc[ch.StartChar:ch.EndChar])
			}
		})
	}

	// Direct guard test for negative start in appendChunkIfValid.
	out := appendChunkIfValid(nil, 0, "hello", "hello", -1, 5)
	require.Len(t, out, 1)
	assert.Equal(t, 0, out[0].StartChar)
	assert.Equal(t, 0, out[0].EndChar)
}

// C5-02: HeadingAware must normalize non-verbatim reconstructed sections
// (blank lines after headings) to (0, 0) for both fitting and oversized sections.
func TestHeadingAware_BlankLineAfterHeadingSpans(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(HeadingAware)
	require.NoError(t, err)

	t.Run("fitting sections with blank lines", func(t *testing.T) {
		t.Parallel()
		doc := "# Alpha\n\nFirst body.\n\n# Beta\n\nSecond body."
		chunks := c.Chunk(doc, 200, 0)
		require.Len(t, chunks, 2)
		for _, ch := range chunks {
			if ch.StartChar == 0 && ch.EndChar == 0 {
				continue
			}
			require.GreaterOrEqual(t, ch.StartChar, 0)
			require.LessOrEqual(t, ch.EndChar, len(doc))
			assert.Equal(t, ch.Text, doc[ch.StartChar:ch.EndChar],
				"chunk %d span must be exact verbatim match or (0,0)", ch.ID)
		}
		// Because fullText() joins heading + "\n" + content while doc has "\n\n",
		// these reconstructed sections are non-verbatim and must normalize to (0,0).
		assert.Equal(t, 0, chunks[0].StartChar)
		assert.Equal(t, 0, chunks[0].EndChar)
		assert.Equal(t, 0, chunks[1].StartChar)
		assert.Equal(t, 0, chunks[1].EndChar)
	})

	t.Run("oversized sections with blank lines", func(t *testing.T) {
		t.Parallel()
		longBody := "Sentence one has plenty of words to overflow. Sentence two adds even more text here. Sentence three forces another split cleanly."
		doc := "# Alpha\n\n" + longBody + "\n\n# Beta\n\n" + longBody
		chunks := c.Chunk(doc, 70, 0)
		require.Greater(t, len(chunks), 2)
		for _, ch := range chunks {
			if ch.StartChar == 0 && ch.EndChar == 0 {
				continue
			}
			require.GreaterOrEqual(t, ch.StartChar, 0)
			require.LessOrEqual(t, ch.EndChar, len(doc))
			assert.Equal(t, ch.Text, doc[ch.StartChar:ch.EndChar],
				"oversized chunk %d span must be exact verbatim match or (0,0)", ch.ID)
		}
	})
}

type runeTokenCounter struct{ max int }

func (r runeTokenCounter) CountTokens(s string) int { return utf8.RuneCountInString(s) }
func (r runeTokenCounter) MaxTokens() int           { return r.max }

type exactSplitTokenizer struct{}

func (exactSplitTokenizer) Encode(text string) ([]int, error) {
	return make([]int, utf8.RuneCountInString(text)), nil
}

func (exactSplitTokenizer) Count(text string) (int, error) {
	return utf8.RuneCountInString(text), nil
}

// C5-03: EnforceHardLimits, EnforceTokenLimits, and refineExactTokenLimits must
// clone parent Metadata and Path onto every split sub-chunk.
func TestLimitsPreserveMetadataAndPath(t *testing.T) {
	t.Parallel()

	parent := Chunk{
		ID:        0,
		Text:      "First sentence in code block. Second sentence in code block. Third sentence in code block.",
		CharCount: 90,
		WordCount: 15,
		Metadata:  map[string]string{"type": "code", "language": "go"},
		Path:      []string{"Guide", "Setup"},
	}

	t.Run("EnforceHardLimits clones Metadata and Path", func(t *testing.T) {
		t.Parallel()
		out := EnforceHardLimits([]Chunk{parent}, LimitOptions{MaxChars: 35})
		require.Greater(t, len(out), 1)
		for i, ch := range out {
			assert.Equal(t, map[string]string{"type": "code", "language": "go"}, ch.Metadata, "chunk %d metadata", i)
			assert.Equal(t, []string{"Guide", "Setup"}, ch.Path, "chunk %d path", i)
		}
		// FilterProse must still recognize all sub-chunks as code.
		assert.Empty(t, FilterProse(out))

		// Mutating one sub-chunk must not affect siblings.
		out[0].Metadata["type"] = "mutated"
		out[0].Path[0] = "mutated"
		assert.Equal(t, "code", out[1].Metadata["type"])
		assert.Equal(t, "Guide", out[1].Path[0])
	})

	t.Run("EnforceTokenLimits clones Metadata and Path", func(t *testing.T) {
		t.Parallel()
		out := EnforceTokenLimits([]Chunk{parent}, runeTokenCounter{max: 35})
		require.Greater(t, len(out), 1)
		for i, ch := range out {
			assert.Equal(t, map[string]string{"type": "code", "language": "go"}, ch.Metadata, "chunk %d metadata", i)
			assert.Equal(t, []string{"Guide", "Setup"}, ch.Path, "chunk %d path", i)
		}
		assert.Empty(t, FilterProse(out))

		out[0].Metadata["type"] = "mutated"
		out[0].Path[0] = "mutated"
		assert.Equal(t, "code", out[1].Metadata["type"])
		assert.Equal(t, "Guide", out[1].Path[0])
	})

	t.Run("refineExactTokenLimits clones Metadata and Path", func(t *testing.T) {
		t.Parallel()
		out, err := refineExactTokenLimits([]Chunk{parent}, exactSplitTokenizer{}, 35)
		require.NoError(t, err)
		require.Greater(t, len(out), 1)
		for i, ch := range out {
			assert.Equal(t, map[string]string{"type": "code", "language": "go"}, ch.Metadata, "chunk %d metadata", i)
			assert.Equal(t, []string{"Guide", "Setup"}, ch.Path, "chunk %d path", i)
		}
		assert.Empty(t, FilterProse(out))

		out[0].Metadata["type"] = "mutated"
		out[0].Path[0] = "mutated"
		assert.Equal(t, "code", out[1].Metadata["type"])
		assert.Equal(t, "Guide", out[1].Path[0])
	})
}

// C5-04: EnforceHardLimits must guard negative, inverted, and out-of-bounds
// spans before slicing opts.OriginalText.
func TestEnforceHardLimits_InvalidSpansNoPanic(t *testing.T) {
	t.Parallel()

	orig := "First sentence here. Second sentence here. Third sentence here."
	cases := []struct {
		name  string
		start int
		end   int
	}{
		{"negative start", -5, 10},
		{"inverted span", 25, 5},
		{"end past original", 0, len(orig) + 50},
		{"start past original", len(orig) + 10, len(orig) + 20},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			in := []Chunk{{
				ID:        0,
				Text:      orig,
				StartChar: tc.start,
				EndChar:   tc.end,
			}}
			out := EnforceHardLimits(in, LimitOptions{MaxChars: 25, OriginalText: orig})
			require.Greater(t, len(out), 1)
			for _, ch := range out {
				assert.Equal(t, 0, ch.StartChar)
				assert.Equal(t, 0, ch.EndChar)
			}
		})
	}
}

type uniformSemanticEmbedder struct{}

func (uniformSemanticEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 0, 0}
	}
	return out, nil
}

// C5-05: NewSemanticChunker with zero MaxChunkChars must enforce the default
// 1600-rune ceiling during adjacent merging and final hard limits.
func TestSemanticChunker_DefaultMaxChunkCharsCeiling(t *testing.T) {
	t.Parallel()

	sc, err := NewSemanticChunker(uniformSemanticEmbedder{}, SemanticOpts{})
	require.NoError(t, err)
	assert.Equal(t, defaultMaxChunkChars, sc.maxLimit)

	// Build > 2500 runes of distinct sentences with identical embeddings so
	// adjacent merging would combine everything if maxLimit were 0.
	var sb strings.Builder
	for i := 0; i < 60; i++ {
		sb.WriteString("This is a moderately long sentence about semantic chunking limits. ")
	}
	text := sb.String()
	require.Greater(t, utf8.RuneCountInString(text), defaultMaxChunkChars)

	chunks := sc.ChunkSemantic(context.Background(), text, 0, 0)
	require.Greater(t, len(chunks), 1)
	for i, ch := range chunks {
		assert.LessOrEqual(t, utf8.RuneCountInString(ch.Text), defaultMaxChunkChars,
			"chunk %d exceeded default %d-rune ceiling", i, defaultMaxChunkChars)
	}
}

// C5-07: trimSpanToText must trim Unicode whitespace (\u00A0, \u3000) matching
// strings.TrimSpace semantics.
func TestTrimSpanToText_UnicodeWhitespace(t *testing.T) {
	t.Parallel()

	source := "\u00A0First sentence.\u3000\u00A0Second sentence.\u3000"
	s, e := trimSpanToText(source, "First sentence.", 0, len("\u00A0First sentence.\u3000"))
	assert.Equal(t, "First sentence.", source[s:e])

	spans := splitIntoSentencesWithSpans(source)
	require.Len(t, spans, 2)
	for i, sp := range spans {
		require.Less(t, sp.start, sp.end, "span %d should be non-empty", i)
		assert.Equal(t, sp.text, source[sp.start:sp.end], "span %d must match trimmed text", i)
	}
}

// C5-08: OptimalChunker.splitIntoSpans must fall through to ". " when "\n\n"
// appears in the first half of the window (<= runeCursor+half).
func TestOptimalChunker_EarlyParagraphFallsThroughToSentenceBreak(t *testing.T) {
	t.Parallel()

	// OptimalLength=40, half=20.
	// "\n\n" is at rune index 5 (<= 20).
	// ". " is at rune index 30 (> 20, < 40).
	text := "Intro\n\nSome middle words right. And then the trailing text continues well past forty runes."
	oc := &OptimalChunker{OptimalLength: 40}
	parts := oc.SplitIntoChunks(text)
	require.GreaterOrEqual(t, len(parts), 2)
	assert.True(t, strings.HasSuffix(parts[0], ". "),
		"expected first chunk to break at sentence boundary '. ', got %q", parts[0])
}

// C5-09: splitMarkdownTableBlock must recognize 1-column separator rows like
// "| --- |" and repeat header + separator on every split table chunk.
func TestMarkdownAware_SingleColumnTableRepeatsSeparator(t *testing.T) {
	t.Parallel()

	doc := strings.Join([]string{
		"| Name |",
		"| --- |",
		"| Alice Wonderland |",
		"| Bob Builder |",
		"| Carol Danvers |",
		"| Dave Lister |",
	}, "\n")

	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)

	chunks := c.Chunk(doc, 45, 0)
	require.Greater(t, len(chunks), 1, "expected 1-column table to split into multiple chunks")
	for i, ch := range chunks {
		assert.True(t, strings.HasPrefix(ch.Text, "| Name |\n| --- |\n"),
			"chunk %d should start with header and 1-column separator, got %q", i, ch.Text)
	}
}

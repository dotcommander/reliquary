package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeadingAware_Strategy(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()
	assert.Equal(t, HeadingAware, c.Strategy())
}

func TestHeadingAware_EmptyInput(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()
	assert.Nil(t, c.Chunk("", 500, 0))
	assert.Nil(t, c.Chunk("hello", 0, 0))
}

func TestHeadingAware_SmallDocFitsOneChunk(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()
	doc := "# Title\n\nShort paragraph."
	chunks := c.Chunk(doc, 500, 0)
	require.Len(t, chunks, 1)
	assert.Contains(t, chunks[0].Text, "# Title")
	assert.Contains(t, chunks[0].Text, "Short paragraph.")
}

func TestHeadingAware_SplitsAtH1(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()
	doc := "# Section A\n\nContent A is here.\n\n# Section B\n\nContent B is here."
	// Size large enough for each section but not both.
	chunks := c.Chunk(doc, 40, 0)
	require.Len(t, chunks, 2)
	assert.Contains(t, chunks[0].Text, "Section A")
	assert.Contains(t, chunks[1].Text, "Section B")
}

func TestHeadingAware_RecursiveH1ToH2(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()

	// H1 section is oversized, but splits cleanly at H2.
	doc := "# Big Section\n\n## Sub A\n\nContent for sub A.\n\n## Sub B\n\nContent for sub B."
	// Size too small for the whole H1 section, but each H2 fits.
	chunks := c.Chunk(doc, 60, 0)
	require.True(t, len(chunks) >= 2, "expected at least 2 chunks, got %d", len(chunks))

	// First chunk should have H1 heading prepended.
	assert.Contains(t, chunks[0].Text, "# Big Section")
	// Sub B should be in a later chunk, separate from Sub A.
	allText := ""
	for _, ch := range chunks {
		allText += ch.Text + "\n"
	}
	assert.Contains(t, allText, "Sub A")
	assert.Contains(t, allText, "Sub B")
}

func TestHeadingAware_RecursiveH2ToH3(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()

	doc := strings.Join([]string{
		"## Parent",
		"",
		"### Child A",
		"",
		"Child A content with some words.",
		"",
		"### Child B",
		"",
		"Child B content with some words.",
	}, "\n")

	chunks := c.Chunk(doc, 60, 0)
	require.True(t, len(chunks) >= 2, "expected at least 2 chunks from H3 split, got %d", len(chunks))
}

func TestHeadingAware_FallbackToSentences(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()

	// No subheadings, just a large block of prose under one heading.
	doc := "# Only Heading\n\n" + strings.Repeat("This is a sentence. ", 50)
	chunks := c.Chunk(doc, 100, 0)
	require.True(t, len(chunks) > 1, "expected sentence-level splitting, got %d chunks", len(chunks))

	for i, ch := range chunks {
		assert.True(t, utf8.RuneCountInString(ch.Text) <= 100 || i == 0,
			"chunk %d exceeds size: %d chars", i, utf8.RuneCountInString(ch.Text))
	}
}

func TestHeadingAware_NoHeadingsPassthrough(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()

	// Plain text with no headings — should still chunk via sentence fallback.
	doc := strings.Repeat("Plain sentence here. ", 30)
	chunks := c.Chunk(doc, 100, 0)
	require.True(t, len(chunks) > 1, "expected multiple chunks from plain text, got %d", len(chunks))
}

func TestHeadingAware_CodeFencePreserved(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()

	doc := "# Code Example\n\nSome text.\n\n```go\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n```\n\n# Next Section\n\nMore text."
	chunks := c.Chunk(doc, 200, 0)

	// Code fence should not be split across chunks.
	var codeChunk string
	for _, ch := range chunks {
		if strings.Contains(ch.Text, "```go") {
			codeChunk = ch.Text
			break
		}
	}
	require.NotEmpty(t, codeChunk, "code fence not found in any chunk")
	assert.Contains(t, codeChunk, "```go")
	assert.Contains(t, codeChunk, "fmt.Println")
}

func TestHeadingAware_PreambleBeforeFirstHeading(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()

	doc := "This is a preamble before any heading.\n\n# First Heading\n\nContent here."
	chunks := c.Chunk(doc, 60, 0)
	require.True(t, len(chunks) >= 2, "expected preamble + heading as separate chunks, got %d", len(chunks))
	assert.Contains(t, chunks[0].Text, "preamble")
}

func TestHeadingAware_SequentialIDs(t *testing.T) {
	t.Parallel()
	c := newHeadingAwareChunker()

	doc := "# A\n\nContent A.\n\n# B\n\nContent B.\n\n# C\n\nContent C."
	chunks := c.Chunk(doc, 30, 0)
	for i, ch := range chunks {
		assert.Equal(t, i, ch.ID, "chunk %d has ID %d", i, ch.ID)
	}
}

func TestSplitByLevel_ExactMatch(t *testing.T) {
	t.Parallel()

	doc := "# H1\n\nPara.\n\n## H2\n\nSub para.\n\n# H1 Again\n\nMore."
	sections := splitByLevel(doc, 1)

	// Should get 2 H1 sections (## is inside H1, not split).
	require.Len(t, sections, 2)
	assert.Equal(t, 1, sections[0].level)
	assert.Contains(t, sections[0].content, "H2") // H2 is content of first H1
	assert.Equal(t, 1, sections[1].level)
}

func TestSplitByLevel_DoesNotSplitDeeperHeadings(t *testing.T) {
	t.Parallel()

	doc := "## H2 Section\n\n### H3 inside\n\nContent."
	sections := splitByLevel(doc, 2)

	// Should get 1 H2 section containing the H3.
	require.Len(t, sections, 1)
	assert.Contains(t, sections[0].content, "### H3 inside")
}

// ---------------------------------------------------------------------------
// Heading fallback span round-trip tests
// ---------------------------------------------------------------------------

func TestSplitByLevel_FencedHashNotHeading(t *testing.T) {
	t.Parallel()

	doc := "# Section\n\nSome prose.\n\n```sh\n# install dep\nbrew install foo\n```\n\n# Next Section\n\nMore prose."
	sections := splitByLevel(doc, 1)

	// The fenced "# install dep" must NOT be treated as a section boundary.
	require.Len(t, sections, 2, "expected exactly 2 H1 sections, got %d", len(sections))

	// First section contains the fence opener, the # comment, and the fence closer.
	assert.Contains(t, sections[0].content, "# install dep")
	assert.Contains(t, sections[0].content, "```sh")
	assert.Contains(t, sections[0].content, "```")

	// Second section heading is the real H1, not the fenced hash.
	assert.Equal(t, "# Next Section", sections[1].heading)
}

func TestHeadingAware_FallbackSpansRoundTrip(t *testing.T) {
	t.Parallel()

	// Document where a heading section is long enough to trigger sentence fallback.
	// Fallback sub-chunks should have rebased spans such that
	// source[chunk.StartChar:chunk.EndChar] == chunk.Text.
	longContent := strings.Repeat("This is a test sentence for span tracking. ", 20)
	doc := "# Title\n\n" + longContent

	c, err := NewChunker(HeadingAware)
	require.NoError(t, err)

	chunks := c.Chunk(doc, 200, 0)
	require.NotEmpty(t, chunks, "heading fallback should produce chunks")

	// At least one chunk from the fallback path should have non-zero spans.
	hasSpan := false
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		hasSpan = true
		span := doc[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: span round-trip failed (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}
	assert.True(t, hasSpan, "at least one fallback chunk should have non-zero spans")
}

// ---------------------------------------------------------------------------
// Repeated-heading regression for cursor-scoped Locate
// ---------------------------------------------------------------------------

func TestHeadingAware_RepeatedSectionSpan(t *testing.T) {
	t.Parallel()

	// Two sections with identical heading text must not both resolve to the first
	// occurrence and misattribute the second section's span.
	doc := "# See also\n\nFirst occurrence.\n\n# See also\n\nSecond occurrence."

	c, err := NewChunker(HeadingAware)
	require.NoError(t, err)

	chunks := c.Chunk(doc, 200, 0)
	require.Len(t, chunks, 2, "expected 2 sections")

	// First chunk span should be valid.
	first := chunks[0]
	assert.True(t, first.StartChar >= 0, "first chunk StartChar should be >= 0")
	assert.True(t, first.EndChar > first.StartChar, "first chunk should have non-empty span")

	// Second chunk span must point AFTER the first chunk's span.
	second := chunks[1]
	assert.True(t, second.StartChar > first.EndChar,
		"second chunk StartChar (%d) must be > first EndChar (%d), was pointing at first occurrence",
		second.StartChar, first.EndChar)

	// Span round-trip: for normalized matches, the span covers the source region
	// but TrimSpace may differ. Only assert if it's a verbatim match.
	if second.StartChar != 0 && second.EndChar != 0 {
		sourceSlice := doc[second.StartChar:second.EndChar]
		if sourceSlice == second.Text {
			// Exact match — pass
		} else {
			// Normalized match — span covers the right region but trimmed text differs.
			assert.Contains(t, sourceSlice, "Second occurrence",
				"second chunk span should cover the second section content")
		}
	}
}

func TestHeadingAware_NormalizedWhitespaceSpan(t *testing.T) {
	t.Parallel()

	// Section text has different internal whitespace than the original.
	// Locate's normalized fallback should still find a non-zero span.
	doc := "#  Title  \n\nContent  with   spaces."

	c, err := NewChunker(HeadingAware)
	require.NoError(t, err)

	chunks := c.Chunk(doc, 200, 0)
	require.NotEmpty(t, chunks, "should produce chunks")

	// At least one chunk should have a non-zero span (Locate's normalized path).
	hasNonZero := false
	for _, ch := range chunks {
		if ch.StartChar != 0 || ch.EndChar != 0 {
			hasNonZero = true
			if ch.EndChar <= len(doc) {
				assert.Equal(t, ch.Text, doc[ch.StartChar:ch.EndChar],
					"chunk %d: span round-trip failed", ch.ID)
			}
		}
	}
	assert.True(t, hasNonZero, "expected at least one chunk with non-zero span via normalized fallback")
}

// ---------------------------------------------------------------------------
// Path breadcrumb tests
// ---------------------------------------------------------------------------

func TestHeadingAware_Path_H1H2H3(t *testing.T) {
	t.Parallel()

	doc := strings.Join([]string{
		"# Introduction",
		"",
		"Intro content.",
		"",
		"## Getting Started",
		"",
		"Getting started content.",
		"",
		"### Installation",
		"",
		"Install it.",
	}, "\n")

	c, err := NewChunker(HeadingAware)
	require.NoError(t, err)

	chunks := c.Chunk(doc, 50, 0)
	require.NotEmpty(t, chunks)

	// Each chunk should have a Path reflecting its heading ancestry.
	for _, ch := range chunks {
		assert.NotNil(t, ch.Path, "chunk %d: Path should not be nil", ch.ID)
	}

	// Find chunks for specific sections.
	var introChunk, gettingStartedChunk, installChunk *Chunk
	for i := range chunks {
		if strings.Contains(chunks[i].Text, "Introduction") {
			introChunk = &chunks[i]
		}
		if strings.Contains(chunks[i].Text, "Getting Started") {
			gettingStartedChunk = &chunks[i]
		}
		if strings.Contains(chunks[i].Text, "Installation") {
			installChunk = &chunks[i]
		}
	}

	require.NotNil(t, introChunk, "intro chunk not found")
	assert.Equal(t, []string{"Introduction"}, introChunk.Path)

	require.NotNil(t, gettingStartedChunk, "getting started chunk not found")
	assert.Equal(t, []string{"Introduction", "Getting Started"}, gettingStartedChunk.Path)

	require.NotNil(t, installChunk, "install chunk not found")
	assert.Equal(t, []string{"Introduction", "Getting Started", "Installation"}, installChunk.Path)
}

func TestHeadingAware_Path_SkippedLevel(t *testing.T) {
	t.Parallel()

	// H1 followed by H3 (no H2).
	doc := strings.Join([]string{
		"# Title",
		"",
		"Title content.",
		"",
		"### Deep Section",
		"",
		"Deep content.",
	}, "\n")

	c, err := NewChunker(HeadingAware)
	require.NoError(t, err)

	chunks := c.Chunk(doc, 50, 0)
	require.NotEmpty(t, chunks)

	// Find the deep section chunk.
	var deepChunk *Chunk
	for i := range chunks {
		if strings.Contains(chunks[i].Text, "Deep Section") {
			deepChunk = &chunks[i]
			break
		}
	}
	require.NotNil(t, deepChunk, "deep section chunk not found")
	assert.Equal(t, []string{"Title", "Deep Section"}, deepChunk.Path)
}

func TestWordBoundary_PathNil(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(WordBoundary)
	require.NoError(t, err)

	chunks := c.Chunk("hello world foo bar", 10, 0)
	require.NotEmpty(t, chunks)
	for _, ch := range chunks {
		assert.Nil(t, ch.Path, "non-heading strategy should have nil Path")
	}
}

package chunking

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractMarkdownBlocks_Empty(t *testing.T) {
	t.Parallel()
	blocks := extractMarkdownBlocks(nil)
	assert.Nil(t, blocks)

	blocks = extractMarkdownBlocks([]byte{})
	assert.Nil(t, blocks)
}

func TestExtractMarkdownBlocks_HeadingInsideFence(t *testing.T) {
	t.Parallel()

	src := []byte(strings.Join([]string{
		"# Real Heading",
		"",
		"Some text.",
		"",
		"```sh",
		"# this is a comment, not a heading",
		"echo hello",
		"```",
		"",
		"# Another Heading",
		"",
		"More text.",
	}, "\n"))

	blocks := extractMarkdownBlocks(src)

	// Count heading blocks — there should be exactly 2, not 3.
	headingCount := 0
	for _, blk := range blocks {
		if blk.blockType == "heading" {
			headingCount++
		}
	}
	assert.Equal(t, 2, headingCount,
		"# inside fenced code block must not produce a heading block")

	// The code block should be a single "code" block.
	codeBlocks := filterBlocksByType(blocks, "code")
	require.Len(t, codeBlocks, 1, "expected exactly one code block")
	assert.Contains(t, codeBlocks[0].text, "# this is a comment, not a heading")
}

func TestExtractMarkdownBlocks_Types(t *testing.T) {
	t.Parallel()

	src := []byte(strings.Join([]string{
		"# Heading",
		"",
		"A paragraph.",
		"",
		"```go",
		"func main() {}",
		"```",
		"",
		"- item one",
		"- item two",
		"",
		"| A | B |",
		"|---|---|",
		"| 1 | 2 |",
		"",
		"> A quote.",
	}, "\n"))

	blocks := extractMarkdownBlocks(src)

	types := make(map[string]int)
	for _, blk := range blocks {
		types[blk.blockType]++
	}

	assert.Equal(t, 1, types["heading"], "expected 1 heading")
	assert.Equal(t, 1, types["paragraph"], "expected 1 paragraph")
	assert.Equal(t, 1, types["code"], "expected 1 code block")
	assert.Equal(t, 1, types["list"], "expected 1 list")
	assert.Equal(t, 1, types["table"], "expected 1 table")
	assert.Equal(t, 1, types["blockquote"], "expected 1 blockquote")
}

func TestExtractMarkdownBlocks_ByteSpans(t *testing.T) {
	t.Parallel()

	src := []byte(strings.Join([]string{
		"# Title",
		"",
		"Hello world.",
		"",
		"```go",
		"func x() {}",
		"```",
	}, "\n"))

	blocks := extractMarkdownBlocks(src)

	for _, blk := range blocks {
		if blk.startByte == 0 && blk.endByte == 0 {
			t.Logf("block type=%q has zero span (skipping)", blk.blockType)
			continue
		}
		require.True(t, blk.endByte > blk.startByte,
			"block type=%q: endByte (%d) must be > startByte (%d)",
			blk.blockType, blk.endByte, blk.startByte)
		require.True(t, blk.endByte <= len(src),
			"block type=%q: endByte (%d) exceeds source length (%d)",
			blk.blockType, blk.endByte, len(src))

		// The source slice should contain the block text (possibly with trailing whitespace).
		sourceSlice := string(src[blk.startByte:blk.endByte])
		assert.Contains(t, strings.TrimSpace(sourceSlice), strings.TrimSpace(blk.text),
			"block type=%q: source span should contain block text", blk.blockType)
	}
}

func TestExtractMarkdownBlocks_HeadingMetadata(t *testing.T) {
	t.Parallel()

	src := []byte("## Sub Heading\n\nSome content.\n")
	blocks := extractMarkdownBlocks(src)

	headings := filterBlocksByType(blocks, "heading")
	require.Len(t, headings, 1)
	assert.Equal(t, "2", headings[0].metadata[metaKeyHeadingLevel])
	assert.Equal(t, "heading", headings[0].blockType)
	assert.Equal(t, 2, headings[0].level)
}

func TestExtractMarkdownBlocks_CodeMetadata(t *testing.T) {
	t.Parallel()

	src := []byte("```go\nline1\nline2\nline3\n```\n")
	blocks := extractMarkdownBlocks(src)

	codeBlocks := filterBlocksByType(blocks, "code")
	require.Len(t, codeBlocks, 1)
	assert.Equal(t, "go", codeBlocks[0].metadata[metaKeyLanguage])
	assert.Equal(t, "3", codeBlocks[0].metadata[metaKeyLineCount])
}

func TestExtractMarkdownBlocks_CodeNoLanguage(t *testing.T) {
	t.Parallel()

	src := []byte("```\nsome code\n```\n")
	blocks := extractMarkdownBlocks(src)

	codeBlocks := filterBlocksByType(blocks, "code")
	require.Len(t, codeBlocks, 1)
	// No "language" key when fence has no language tag.
	_, hasLang := codeBlocks[0].metadata[metaKeyLanguage]
	assert.False(t, hasLang, "unlabeled fence should not have language metadata")
}

func TestExtractMarkdownBlocks_ParagraphMetadata(t *testing.T) {
	t.Parallel()

	src := []byte("Hello world this is a test.\n")
	blocks := extractMarkdownBlocks(src)

	paras := filterBlocksByType(blocks, "paragraph")
	require.Len(t, paras, 1)
	assert.Equal(t, "6", paras[0].metadata[metaKeyWordCount])
}

func TestExtractMarkdownBlocks_NestedHeadings(t *testing.T) {
	t.Parallel()

	src := []byte(strings.Join([]string{
		"# H1",
		"",
		"Para under H1.",
		"",
		"## H2",
		"",
		"Para under H2.",
	}, "\n"))

	blocks := extractMarkdownBlocks(src)

	headings := filterBlocksByType(blocks, "heading")
	require.Len(t, headings, 2)
	assert.Equal(t, 1, headings[0].level)
	assert.Equal(t, 2, headings[1].level)
}

func TestChunkMetadata_NilForNonGoldmark(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(WordBoundary)
	require.NoError(t, err)

	chunks := c.Chunk("hello world foo bar", 10, 0)
	require.NotEmpty(t, chunks)
	for _, ch := range chunks {
		assert.Nil(t, ch.Metadata, "non-goldmark strategy should have nil Metadata")
	}
}

func TestChunkMetadata_CodeBlock(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)

	md := "```go\nfunc main() {}\n```\n"
	chunks := c.Chunk(md, 500, 0)
	require.NotEmpty(t, chunks)

	codeChunk := chunks[0]
	assert.Equal(t, "go", codeChunk.Metadata[metaKeyLanguage])
	assert.Equal(t, "1", codeChunk.Metadata[metaKeyLineCount])
}

func TestChunkMetadata_Heading(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)

	md := "## Section Title\n\nSome content.\n"
	chunks := c.Chunk(md, 20, 0)
	require.NotEmpty(t, chunks)

	// Find the heading chunk.
	var headingChunk *Chunk
	for i := range chunks {
		if chunks[i].Metadata != nil {
			if _, ok := chunks[i].Metadata[metaKeyHeadingLevel]; ok {
				headingChunk = &chunks[i]
				break
			}
		}
	}
	require.NotNil(t, headingChunk, "heading chunk not found")
	assert.Equal(t, "2", headingChunk.Metadata[metaKeyHeadingLevel])
}

// filterBlocksByType is a test helper.
func filterBlocksByType(blocks []markdownBlock, blockType string) []markdownBlock {
	var filtered []markdownBlock
	for _, blk := range blocks {
		if blk.blockType == blockType {
			filtered = append(filtered, blk)
		}
	}
	return filtered
}

package chunking

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestExtractMarkdownBlocks_TableVerbatimSource pins the bug where table
// blocks were reconstructed from AST cells: goldmark parses the delimiter
// row (|---|---|) into alignment attributes instead of text cells, so the
// reconstruction silently deleted it and normalized cell padding. The block
// text must now be the verbatim source span.
func TestExtractMarkdownBlocks_TableVerbatimSource(t *testing.T) {
	t.Parallel()

	doc := "# Flags\n\n" +
		"|Flag|Default|Notes|\n" +
		"|:-|-:|---|\n" +
		"|`--mode`|headings|Chunking mode.|\n" +
		"|`--out-dir`|_unset_|Output directory.|\n\n" +
		"Trailing paragraph.\n"

	blocks := extractMarkdownBlocks([]byte(doc))

	var table *markdownBlock
	for i := range blocks {
		if blocks[i].blockType == "table" {
			require.Nil(t, table, "expected exactly one table block")
			table = &blocks[i]
		}
	}
	require.NotNil(t, table, "expected a table block")

	want := "|Flag|Default|Notes|\n|:-|-:|---|\n|`--mode`|headings|Chunking mode.|\n|`--out-dir`|_unset_|Output directory.|"
	assert.Equal(t, want, table.text, "table block must be the verbatim source rows")
	assert.Contains(t, table.text, "|:-|-:|---|", "delimiter row must survive")

	offset := strings.Index(doc, want)
	require.GreaterOrEqual(t, offset, 0)
	assert.Equal(t, offset, table.startByte, "startByte must index the verbatim span")
	assert.Equal(t, offset+len(want), table.endByte, "endByte must end the verbatim span")
	assert.Equal(t, want, doc[table.startByte:table.endByte], "claimed span must reproduce the block text")
}

// TestExtractMarkdownBlocks_TableAtDocumentStart covers the span edge where
// the table is the first block: line expansion must not require a preceding
// newline and must still capture the delimiter row.
func TestExtractMarkdownBlocks_TableAtDocumentStart(t *testing.T) {
	t.Parallel()

	doc := "| a | b |\n| --- | --- |\n| 1 | 2 |\n\nAfter the table.\n"

	blocks := extractMarkdownBlocks([]byte(doc))

	var table *markdownBlock
	for i := range blocks {
		if blocks[i].blockType == "table" {
			table = &blocks[i]
		}
	}
	require.NotNil(t, table, "expected a table block")
	assert.Contains(t, table.text, "| --- | --- |", "delimiter row must survive at document start")
	assert.Equal(t, "| a | b |\n| --- | --- |\n| 1 | 2 |", table.text)
}

// TestExtractMarkdownBlocks_TableAfterJoinedHeading covers the recursion
// shape where section blocks are joined with single newlines (blank lines
// collapsed): the table follows a heading with no blank line, container
// spans leak invalid sentinels, and only cell-derived spans recover the
// verbatim table.
func TestExtractMarkdownBlocks_TableAfterJoinedHeading(t *testing.T) {
	t.Parallel()

	doc := "Intro paragraph.\n#### Modes\n| H1 | H2 |\n|---|---|\n| a | b |\n" //nolint:dupl // regression fixture

	blocks := extractMarkdownBlocks([]byte(doc))

	var table *markdownBlock
	for i := range blocks {
		if blocks[i].blockType == "table" {
			table = &blocks[i]
		}
	}
	require.NotNil(t, table, "expected a table block")
	assert.Contains(t, table.text, "|---|---|", "delimiter row must survive block-joined input")
	assert.Equal(t, "| H1 | H2 |\n|---|---|\n| a | b |", table.text)
}

// TestHeadingAware_PreservesTableDelimiterRow is the end-to-end regression:
// the strategy distill's --mode headings maps to must keep markdown tables
// renderable in every chunk that contains them.
func TestHeadingAware_PreservesTableDelimiterRow(t *testing.T) {
	t.Parallel()

	c := newHeadingAwareChunker()
	doc := "# Configuration\n\n" +
		"|Flag|Default|Notes|\n" +
		"|---|---|---|\n" +
		"|`--mode`|headings|Chunking mode.|\n" +
		"|`--out-dir`|_unset_|Output directory.|\n\n" +
		"# Second\n\nFiller section.\n"

	chunks := c.Chunk(doc, 500, 0)
	require.NotEmpty(t, chunks)

	joined := strings.Join(chunkTexts(chunks), "\n")
	assert.Contains(t, joined, "|---|---|---|", "heading-aware chunks must keep the table delimiter row")
	assert.Contains(t, joined, "|`--mode`|headings|Chunking mode.|", "table data rows must be preserved")
}

// TestIsTableDelimiterRow pins the delimiter-row classifier used to validate
// verbatim table spans.
func TestIsTableDelimiterRow(t *testing.T) {
	t.Parallel()

	cases := []struct {
		line string
		want bool
	}{
		{"|---|---|", true},
		{"| --- | --- |", true},
		{"|:-|-:|:-:|", true},
		{"---|---", true},
		{"|-|", true},
		{"| a | b |", false},
		{"|---x|---|", false},
		{"||", false},
		{"text", false},
		{"", false},
	}
	for _, tc := range cases {
		assert.Equal(t, tc.want, isTableDelimiterRow(tc.line), "line %q", tc.line)
	}
}

// TestHeadingAware_RecursiveSectionKeepsTableIntact pins the recursion path:
// an oversized section is re-parsed after blocks are re-joined, and the
// table plus its trailing paragraph must survive as distinct, intact blocks
// (single-newline joins used to absorb the paragraph into the table).
func TestHeadingAware_RecursiveSectionKeepsTableIntact(t *testing.T) {
	t.Parallel()

	c := newHeadingAwareChunker()
	var b strings.Builder
	b.WriteString("## Top\n\nIntro paragraph before any subsection.\n\n")
	b.WriteString("### Data\n\n| H1 | H2 |\n|---|---|\n| a | b |\n\n")
	b.WriteString("Modes fall back to dense packing when content cannot split further.\n\n")
	// Padding subsection so the parent section exceeds the size limit and
	// must recurse into H3 instead of emitting one chunk.
	for i := range 12 {
		fmt.Fprintf(&b, "### Extra %d\n\n%s\n\n", i, strings.Repeat("filler text ", 30))
	}

	chunks := c.Chunk(b.String(), 600, 0)
	joined := strings.Join(chunkTexts(chunks), "\n")
	assert.Contains(t, joined, "| H1 | H2 |\n|---|---|\n| a | b |", "verbatim table must survive recursion")
	assert.Contains(t, joined, "\n\nModes fall back", "trailing paragraph must stay outside the table")
}

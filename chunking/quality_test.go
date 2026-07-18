package chunking

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Quality fixtures: deterministic corpus tests guarding spans, budgets,
// ordering, overlap, markdown fences, and Unicode.
// ---------------------------------------------------------------------------

// corpusEntry is a reusable test input for quality checks.
type corpusEntry struct {
	name     string
	text     string
	strategy Strategy
	size     int
	overlap  int
}

// standardCorpus returns a fixed set of inputs covering common edge cases.
func standardCorpus() []corpusEntry {
	return []corpusEntry{
		{
			name:     "plain_english",
			text:     "The quick brown fox jumps over the lazy dog. A second sentence follows here. Third sentence for good measure.",
			strategy: SmartBoundary,
			size:     50,
			overlap:  0,
		},
		{
			name:     "repeated_text",
			text:     strings.Repeat("Hello world. ", 20),
			strategy: SentenceBoundary,
			size:     60,
			overlap:  0,
		},
		{
			name:     "paragraphs",
			text:     "First paragraph here.\n\nSecond paragraph with more text.\n\nThird and final paragraph.",
			strategy: ParagraphAware,
			size:     100,
			overlap:  0,
		},
		{
			name:     "markdown_code_fence",
			text:     "Some intro text.\n\n```go\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n```\n\nAfter code.",
			strategy: MarkdownAware,
			size:     200,
			overlap:  0,
		},
		{
			name:     "unicode_emoji",
			text:     "Hello 🌍 world 🌙 test. Another sentence with émojis 🎉 and such.",
			strategy: SmartBoundary,
			size:     40,
			overlap:  0,
		},
		{
			name:     "heading_sections",
			text:     "# Section One\nContent for section one.\n\n# Section Two\nContent for section two here.\n\n# Section Three\nFinal content.",
			strategy: HeadingAware,
			size:     200,
			overlap:  0,
		},
		{
			name:     "hard_cut_unicode",
			text:     "αβγδεζηθικλμνξοπρστυφχψω",
			strategy: HardCut,
			size:     5,
			overlap:  0,
		},
		{
			name:     "word_boundary_mixed",
			text:     "alpha beta gamma delta epsilon zeta eta theta iota kappa",
			strategy: WordBoundary,
			size:     20,
			overlap:  0,
		},
	}
}

// TestQuality_SpanRoundTrip verifies that chunk text matches the source slice
// for all strategies in the standard corpus.
func TestQuality_SpanRoundTrip(t *testing.T) {
	t.Parallel()

	for _, tc := range standardCorpus() {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			c, err := NewChunker(tc.strategy)
			require.NoError(t, err)

			chunks := c.Chunk(tc.text, tc.size, tc.overlap)
			require.NotEmpty(t, chunks, "expected at least one chunk")

			for _, ch := range chunks {
				if ch.StartChar == 0 && ch.EndChar == 0 {
					continue // sub-chunk from split, no span
				}
				if ch.StartChar < 0 || ch.EndChar > len(tc.text) {
					t.Errorf("span out of bounds: [%d, %d) for text length %d",
						ch.StartChar, ch.EndChar, len(tc.text))
					continue
				}
				span := tc.text[ch.StartChar:ch.EndChar]
				if span != ch.Text {
					t.Errorf("span round-trip failed: span=%q text=%q", span, ch.Text)
				}
			}
		})
	}
}

// TestQuality_Ordering verifies that chunk IDs are sequential and text order
// is preserved.
func TestQuality_Ordering(t *testing.T) {
	t.Parallel()

	for _, tc := range standardCorpus() {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			c, err := NewChunker(tc.strategy)
			require.NoError(t, err)

			chunks := c.Chunk(tc.text, tc.size, tc.overlap)
			require.NotEmpty(t, chunks)

			for i, ch := range chunks {
				assert.Equal(t, i, ch.ID, "chunk ID not sequential at index %d", i)
			}

			// Concatenated text should contain all significant words from original.
			var allText strings.Builder
			for _, ch := range chunks {
				allText.WriteString(ch.Text)
				allText.WriteString(" ")
			}
			combined := allText.String()
			for _, word := range strings.Fields(tc.text) {
				// Skip word check when the strategy can split mid-word and the word
				// is longer than the chunk size.
				if tc.strategy == HardCut && utf8.RuneCountInString(word) > tc.size {
					continue
				}
				if !strings.Contains(combined, word) {
					t.Errorf("word %q from original not found in any chunk", word)
				}
			}
		})
	}
}

// TestQuality_OverlapInflation checks that overlap doesn't cause total
// character count to more than double (indicating a runaway overlap bug).
func TestQuality_OverlapInflation(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("The quick brown fox jumps. ", 20)
	origChars := utf8.RuneCountInString(text)

	for _, strategy := range []Strategy{SmartBoundary, SentenceBoundary, WordBoundary, HardCut} {
		t.Run(string(strategy), func(t *testing.T) {
			t.Parallel()

			c, err := NewChunker(strategy)
			require.NoError(t, err)

			chunks := c.Chunk(text, 50, 10)
			require.NotEmpty(t, chunks)

			totalChars := 0
			for _, ch := range chunks {
				totalChars += utf8.RuneCountInString(ch.Text)
			}

			// Overlap can at most double the total. 2.5x safety margin.
			if totalChars > origChars*5/2 {
				t.Errorf("overlap inflation too high: original=%d total=%d (%.1fx)",
					origChars, totalChars, float64(totalChars)/float64(origChars))
			}
		})
	}
}

// TestQuality_MarkdownFenceIntegrity verifies that code fences are not split
// mid-block by the markdown-aware chunker.
func TestQuality_MarkdownFenceIntegrity(t *testing.T) {
	t.Parallel()

	text := "# Title\n\nSome text before.\n\n```python\ndef hello():\n    print('world')\n```\n\nText after."
	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)

	chunks := c.Chunk(text, 200, 0)
	require.NotEmpty(t, chunks)

	// No chunk should contain an unclosed fence.
	for _, ch := range chunks {
		fenceCount := strings.Count(ch.Text, "```")
		assert.False(t, fenceCount%2 != 0,
			"chunk %d has unbalanced code fences (%d): %q", ch.ID, fenceCount, ch.Text)
	}
}

// TestQuality_UnicodeByteSpans verifies that byte offsets correctly handle
// multi-byte UTF-8 characters.
func TestQuality_UnicodeByteSpans(t *testing.T) {
	t.Parallel()

	// Each CJK character is 3 bytes, each emoji is 4 bytes.
	text := "你好世界 test 测试数据"
	c, err := NewChunker(HardCut)
	require.NoError(t, err)

	chunks := c.Chunk(text, 5, 0)
	require.NotEmpty(t, chunks)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		// Span must be valid UTF-8 boundaries.
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"byte span mismatch for chunk %d: got %q want %q", ch.ID, span, ch.Text)

		// Verify valid UTF-8.
		assert.True(t, utf8.ValidString(span), "span is not valid UTF-8")
	}
}

// TestQuality_TokenBudgetEnforcement verifies that EnforceTokenLimits
// produces chunks within budget.
func TestQuality_TokenBudgetEnforcement(t *testing.T) {
	t.Parallel()

	text := "alpha beta gamma delta. epsilon zeta eta theta. iota kappa lambda mu."
	c, _ := NewChunker(SentenceBoundary)
	chunks := c.Chunk(text, 1000, 0)

	tc := fakeTokenCounter{maxTokens: 3}
	result := EnforceTokenLimits(chunks, tc)

	for _, ch := range result {
		tokCount := tc.CountTokens(ch.Text)
		assert.LessOrEqual(t, tokCount, tc.MaxTokens(),
			"chunk %d exceeds token budget: %d > %d, text %q",
			ch.ID, tokCount, tc.MaxTokens(), ch.Text)
	}

	// All words from original should appear in output.
	var allText strings.Builder
	for _, ch := range result {
		allText.WriteString(ch.Text)
		allText.WriteString(" ")
	}
	combined := allText.String()
	for _, word := range strings.Fields(text) {
		assert.Contains(t, combined, word, "word %q lost after token limiting", word)
	}
}

// ---------------------------------------------------------------------------
// Table-aware markdown chunking tests
// ---------------------------------------------------------------------------

// tableFixture builds a simple markdown table with the given number of body rows.
func tableFixture(numRows int) string {
	var buf strings.Builder
	buf.WriteString("| Name | Type | Value |\n")
	buf.WriteString("|------|------|--------|\n")
	for i := range numRows {
		fmt.Fprintf(&buf, "| item-%d | type-%d | %s |\n", i, i, strings.Repeat("x", 30))
	}
	return buf.String()
}

func TestMarkdownAware_TableChunksRepeatHeader(t *testing.T) {
	t.Parallel()

	// Build a table large enough to require splitting.
	table := tableFixture(30)
	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)
	chunks := c.Chunk(table, 200, 0)
	require.Greater(t, len(chunks), 1, "expected multiple chunks from oversized table")

	// Goldmark reformats tables: separator row is stripped, cells are normalized.
	// The header row (first row) should appear in every chunk.
	header := "| Name | Type | Value |"
	for _, ch := range chunks {
		lines := strings.Split(ch.Text, "\n")
		require.NotEmpty(t, lines, "table chunk should have lines: %q", ch.Text)
		assert.Equal(t, header, lines[0], "first line should be header row in chunk %d", ch.ID)
	}
}

func TestMarkdownAware_TableChunksPreserveRows(t *testing.T) {
	t.Parallel()

	table := tableFixture(20)
	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)
	chunks := c.Chunk(table, 300, 0)

	// Collect all body rows from all chunks (skip header line).
	var allRows []string
	for _, ch := range chunks {
		lines := strings.Split(ch.Text, "\n")
		for i, line := range lines {
			if i == 0 {
				continue // skip header row
			}
			if strings.TrimSpace(line) != "" {
				allRows = append(allRows, line)
			}
		}
	}

	// Should have all 20 body rows in order.
	assert.Len(t, allRows, 20, "all body rows should be preserved")
	for i, row := range allRows {
		assert.Contains(t, row, fmt.Sprintf("item-%d", i),
			"row %d out of order or missing: %q", i, row)
	}
}

func TestMarkdownAware_TableChunksRespectSize(t *testing.T) {
	t.Parallel()

	// Use rows that each fit comfortably with the header.
	table := tableFixture(40)
	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)
	chunks := c.Chunk(table, 200, 0)
	require.NotEmpty(t, chunks)

	// When each row + header fits, all chunks should be within budget.
	for _, ch := range chunks {
		runeCount := utf8.RuneCountInString(ch.Text)
		// Allow some tolerance for EnforceHardLimits overhead.
		assert.LessOrEqual(t, runeCount, 250,
			"table chunk %d exceeds budget: %d runes", ch.ID, runeCount)
	}
}

func TestMarkdownAware_TableChunksUseZeroSpanForSyntheticHeaders(t *testing.T) {
	t.Parallel()

	table := tableFixture(30)
	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)
	chunks := c.Chunk(table, 200, 0)

	// Table chunks use buildChunk which sets zero spans.
	for _, ch := range chunks {
		assert.Equal(t, 0, ch.StartChar, "table chunk should have zero StartChar")
		assert.Equal(t, 0, ch.EndChar, "table chunk should have zero EndChar")
	}
}

func TestMarkdownAware_NonTablePipeTextUnchanged(t *testing.T) {
	t.Parallel()

	// Prose with pipes but no separator row — should not be table-split.
	text := "This is some text with a | pipe character. And another | here.\n" +
		"More text without any separator."

	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)
	chunks := c.Chunk(text, 500, 0)
	require.Len(t, chunks, 1, "prose with pipes should be one chunk")
	assert.Equal(t, text, chunks[0].Text)
}

func TestMarkdownAware_TableInsideFenceUnchanged(t *testing.T) {
	t.Parallel()

	// A fenced code block that contains pipe-table-like text.
	text := "```markdown\n" +
		"| A | B |\n" +
		"|---|---|\n" +
		"| 1 | 2 |\n" +
		"```"

	c, err := NewChunker(MarkdownAware)
	require.NoError(t, err)
	chunks := c.Chunk(text, 500, 0)
	require.Len(t, chunks, 1, "fenced table should be one code block chunk")
	// Content should be the fenced block as-is.
	assert.Contains(t, chunks[0].Text, "```markdown")
	assert.Contains(t, chunks[0].Text, "| A | B |")
}

// TestQuality_EmptyInputGraceful verifies empty/whitespace inputs don't panic.
func TestQuality_EmptyInputGraceful(t *testing.T) {
	t.Parallel()

	for _, strategy := range []Strategy{HardCut, SmartBoundary, SentenceBoundary, WordBoundary, ParagraphAware, MarkdownAware, HeadingAware} {
		t.Run(string(strategy), func(t *testing.T) {
			t.Parallel()

			c, err := NewChunker(strategy)
			require.NoError(t, err)

			chunks := c.Chunk("", 100, 0)
			assert.Nil(t, chunks)

			chunks = c.Chunk("   \n\t  ", 100, 0)
			// Should return nil or empty (no meaningful chunks).
			for _, ch := range chunks {
				assert.NotEqual(t, "", strings.TrimSpace(ch.Text), "whitespace-only chunk produced")
			}
		})
	}
}

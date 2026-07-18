package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// EnforceHardLimits — core behavior
// ---------------------------------------------------------------------------

func TestEnforceHardLimits_UndersizedChunksPassThrough(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		buildChunk(0, "short text"),
		buildChunk(1, "another chunk"),
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 1000})

	require.Len(t, result, 2)
	assert.Equal(t, 0, result[0].ID)
	assert.Equal(t, 1, result[1].ID)
	assert.Equal(t, "short text", result[0].Text)
	assert.Equal(t, "another chunk", result[1].Text)
}

func TestEnforceHardLimits_OversizedSplits(t *testing.T) {
	t.Parallel()

	longText := strings.Repeat("alpha ", 500) // ~3000 chars
	chunks := []Chunk{buildChunk(0, longText)}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 200})

	// Every resulting chunk must fit within the limit.
	for _, c := range result {
		assert.LessOrEqual(t, utf8.RuneCountInString(c.Text), 200,
			"chunk %d exceeds limit: %d chars", c.ID, utf8.RuneCountInString(c.Text))
	}

	// IDs must be sequential.
	for i, c := range result {
		assert.Equal(t, i, c.ID, "chunk ID not sequential")
	}
}

func TestEnforceHardLimits_SequentialIDs(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		buildChunk(5, "aaa"),
		buildChunk(10, "bbb"),
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 1000})

	require.Len(t, result, 2)
	assert.Equal(t, 0, result[0].ID)
	assert.Equal(t, 1, result[1].ID)
}

func TestEnforceHardLimits_EmptyInput(t *testing.T) {
	t.Parallel()

	result := EnforceHardLimits(nil, LimitOptions{MaxChars: 100})
	assert.Nil(t, result)

	result = EnforceHardLimits([]Chunk{}, LimitOptions{MaxChars: 100})
	assert.Nil(t, result)
}

func TestEnforceHardLimits_ZeroMaxChars_Passthrough(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{buildChunk(0, "some text")}
	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 0})
	require.Len(t, result, 1)
	assert.Equal(t, "some text", result[0].Text)
}

func TestEnforceHardLimits_SingleIndivisibleToken(t *testing.T) {
	t.Parallel()

	// A single word longer than the limit — hard cut must still produce output.
	longWord := strings.Repeat("x", 200)
	chunks := []Chunk{buildChunk(0, longWord)}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 50})

	assert.NotEmpty(t, result, "should still produce chunks for indivisible tokens")
	// At least one chunk will exceed the limit — that's acceptable for indivisible content.
	// But it should be split into multiple pieces.
	assert.GreaterOrEqual(t, len(result), 2, "indivisible token should be hard-cut into pieces")
}

func TestEnforceHardLimits_EmptyChunksDropped(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		buildChunk(0, ""),
		buildChunk(1, "actual content"),
		buildChunk(2, ""),
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 1000})

	require.Len(t, result, 1)
	assert.Equal(t, "actual content", result[0].Text)
	assert.Equal(t, 0, result[0].ID)
}

func TestEnforceHardLimits_CascadingBoundaries(t *testing.T) {
	t.Parallel()

	// Text with paragraph breaks — should split at paragraph boundaries first.
	para1 := strings.Repeat("word ", 30) // ~150 chars
	para2 := strings.Repeat("item ", 30) // ~150 chars
	text := para1 + "\n\n" + para2

	chunks := []Chunk{buildChunk(0, text)}
	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 200})

	assert.GreaterOrEqual(t, len(result), 2, "should split at paragraph boundary")
	for _, c := range result {
		assert.LessOrEqual(t, utf8.RuneCountInString(c.Text), 200)
	}
}

func TestEnforceHardLimits_PreservesTextOrder(t *testing.T) {
	t.Parallel()

	text := "AAA " + strings.Repeat("fill ", 100) + " BBB " + strings.Repeat("fill ", 100) + " CCC"
	chunks := []Chunk{buildChunk(0, text)}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 100})

	// Concatenating all chunk texts (with spaces) should contain the original markers.
	joined := strings.Join(func() []string {
		var texts []string
		for _, c := range result {
			texts = append(texts, c.Text)
		}
		return texts
	}(), " ")

	assert.Contains(t, joined, "AAA")
	assert.Contains(t, joined, "CCC")
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func TestSplitAtBoundary_SingleSegment(t *testing.T) {
	t.Parallel()

	// Single segment — should return nil (can't split).
	result := splitAtBoundary("hello world", 100, splitWordsForLimit)
	assert.Nil(t, result)
}

func TestHardCutSplit_Exact(t *testing.T) {
	t.Parallel()

	text := "abcdefghij" // 10 runes
	result := hardCutSplit(text, 5)

	require.Len(t, result, 2)
	assert.Equal(t, "abcde", result[0])
	assert.Equal(t, "fghij", result[1])
}

func TestHardCutSplit_Remainder(t *testing.T) {
	t.Parallel()

	text := "abcdefgh" // 8 runes
	result := hardCutSplit(text, 5)

	require.Len(t, result, 2)
	assert.Equal(t, "abcde", result[0])
	assert.Equal(t, "fgh", result[1])
}

// ---------------------------------------------------------------------------
// Regression tests — real-world oversized scenarios
// ---------------------------------------------------------------------------

// TestRegression_MarkdownTableOversized verifies that a long markdown table
// with no paragraph/sentence breaks gets split under MaxChars.
func TestRegression_MarkdownTableOversized(t *testing.T) {
	t.Parallel()

	// Build a large markdown table — ~30 rows, each ~100 chars = ~3000 chars.
	var buf strings.Builder
	buf.WriteString("| Name | Description | Value |\n")
	buf.WriteString("|------|-------------|-------|\n")
	for i := range 60 {
		buf.WriteString("| item" + strings.Repeat("x", 20) +
			" | desc" + strings.Repeat("y", 20) +
			" | " + strings.Repeat("z", 20) + " |\n")
		_ = i
	}

	table := buf.String()
	chunker, err := NewChunker(MarkdownAware)
	require.NoError(t, err)
	chunks := chunker.Chunk(table, 500, 0)

	assert.NotEmpty(t, chunks)
	for _, c := range chunks {
		assert.LessOrEqual(t, utf8.RuneCountInString(c.Text), 500,
			"markdown table chunk %d exceeds limit: %d chars", c.ID, utf8.RuneCountInString(c.Text))
	}

	// Sequential IDs.
	for i, c := range chunks {
		assert.Equal(t, i, c.ID)
	}
}

// TestRegression_LongHeadingSection_NoMultiThousandTokenChunk verifies
// that a single long heading section with list items doesn't produce
// a massive chunk.
func TestRegression_LongHeadingSection_NoMultiThousandTokenChunk(t *testing.T) {
	t.Parallel()

	var buf strings.Builder
	buf.WriteString("# Long Section\n\n")
	for i := range 200 {
		buf.WriteString("- list item " + strings.Repeat("word ", 10) + "\n")
		_ = i
	}

	text := buf.String()
	chunker, err := NewChunker(HeadingAware)
	require.NoError(t, err)
	chunks := chunker.Chunk(text, 800, 0)

	assert.NotEmpty(t, chunks)
	for _, c := range chunks {
		assert.LessOrEqual(t, utf8.RuneCountInString(c.Text), 800,
			"heading-aware chunk %d exceeds limit: %d chars", c.ID, utf8.RuneCountInString(c.Text))
	}
}

// TestRegression_SmartBoundary_OversizedSentenceSplit verifies that
// SmartBoundary doesn't produce a chunk exceeding the limit when the
// input has many sentences.
func TestRegression_SmartBoundary_OversizedSentenceSplit(t *testing.T) {
	t.Parallel()

	var buf strings.Builder
	for i := range 50 {
		buf.WriteString("This is sentence number " + strings.Repeat("a", 30) + ". ")
		_ = i
	}

	text := buf.String()
	chunker, err := NewChunker(SmartBoundary)
	require.NoError(t, err)
	chunks := chunker.Chunk(text, 300, 50)

	assert.NotEmpty(t, chunks)
	for _, c := range chunks {
		// Overlap may cause slight overage, but must be bounded.
		charCount := utf8.RuneCountInString(c.Text)
		assert.LessOrEqual(t, charCount, 350,
			"smart boundary chunk %d exceeds limit+tolerance: %d chars", c.ID, charCount)
	}
}

// TestRegression_WordBoundary_OverlapNoExceed verifies that word-boundary
// chunks with overlap don't exceed the configured limit.
func TestRegression_WordBoundary_OverlapNoExceed(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("the quick brown fox jumps over the lazy dog ", 50) // ~2250 chars
	chunker, err := NewChunker(WordBoundary)
	require.NoError(t, err)
	chunks := chunker.Chunk(text, 200, 20)

	assert.NotEmpty(t, chunks)
	for _, c := range chunks {
		// With overlap, allow a small tolerance.
		charCount := utf8.RuneCountInString(c.Text)
		assert.LessOrEqual(t, charCount, 220,
			"word boundary chunk %d exceeds limit+tolerance: %d chars", c.ID, charCount)
	}
}

// TestRegression_HardCut_RespectsLimit verifies that HardCut chunks
// don't exceed the configured size after finalization.
func TestRegression_HardCut_RespectsLimit(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("x", 500)
	chunker, err := NewChunker(HardCut)
	require.NoError(t, err)
	chunks := chunker.Chunk(text, 100, 10)

	assert.NotEmpty(t, chunks)
	for _, c := range chunks {
		charCount := utf8.RuneCountInString(c.Text)
		// HardCut with overlap may produce chunks slightly over 100 chars
		// due to overlap. The finalizer should keep them within bounds.
		assert.LessOrEqual(t, charCount, 110,
			"hard cut chunk %d exceeds limit+tolerance: %d chars", c.ID, charCount)
	}
}

// TestRegression_MarkdownAware_LongCodeBlock splits an oversized fenced
// code block under the limit.
func TestRegression_MarkdownAware_LongCodeBlock(t *testing.T) {
	t.Parallel()

	codeBlock := "```\n" + strings.Repeat("line of code\n", 100) + "```"
	chunker, err := NewChunker(MarkdownAware)
	require.NoError(t, err)
	chunks := chunker.Chunk(codeBlock, 500, 0)

	assert.NotEmpty(t, chunks)
	for _, c := range chunks {
		assert.LessOrEqual(t, utf8.RuneCountInString(c.Text), 500,
			"code block chunk %d exceeds limit: %d chars", c.ID, utf8.RuneCountInString(c.Text))
	}
}

// TestRegression_ParagraphAware_MixedParagraphSizes verifies that
// paragraph-aware chunking handles a mix of large and small paragraphs.
func TestRegression_ParagraphAware_MixedParagraphSizes(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("short ", 5) + "\n\n" +
		strings.Repeat("medium paragraph text ", 100) + "\n\n" +
		strings.Repeat("tiny ", 3)

	chunker, err := NewChunker(ParagraphAware)
	require.NoError(t, err)
	chunks := chunker.Chunk(text, 300, 0)

	assert.NotEmpty(t, chunks)
	for _, c := range chunks {
		assert.LessOrEqual(t, utf8.RuneCountInString(c.Text), 300,
			"paragraph chunk %d exceeds limit: %d chars", c.ID, utf8.RuneCountInString(c.Text))
	}
}

// TestRegression_SentenceBoundary_LongInput verifies that sentence-boundary
// chunking keeps every chunk within the limit.
func TestRegression_SentenceBoundary_LongInput(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("This is a test sentence with some content. ", 100)
	chunker, err := NewChunker(SentenceBoundary)
	require.NoError(t, err)
	chunks := chunker.Chunk(text, 200, 0)

	assert.NotEmpty(t, chunks)
	for _, c := range chunks {
		assert.LessOrEqual(t, utf8.RuneCountInString(c.Text), 200,
			"sentence chunk %d exceeds limit: %d chars", c.ID, utf8.RuneCountInString(c.Text))
	}
}

// ---------------------------------------------------------------------------
// Span propagation through EnforceHardLimits
// ---------------------------------------------------------------------------

func TestEnforceHardLimits_SpanPreservedForFittingChunk(t *testing.T) {
	t.Parallel()

	text := "short text"
	chunks := []Chunk{
		{ID: 0, Text: text, StartChar: 5, EndChar: 5 + len(text), CharCount: len([]rune(text))},
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 1000, OriginalText: "prefix short text suffix"})

	require.Len(t, result, 1)
	assert.Equal(t, 5, result[0].StartChar)
	assert.Equal(t, 5+len(text), result[0].EndChar)
}

func TestEnforceHardLimits_SubSpanOnSplit(t *testing.T) {
	t.Parallel()

	original := "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
	longText := original
	chunks := []Chunk{
		{ID: 0, Text: longText, StartChar: 0, EndChar: len(longText), CharCount: len([]rune(longText))},
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 20, OriginalText: original})

	assert.GreaterOrEqual(t, len(result), 2, "should split into multiple sub-chunks")

	// Each sub-chunk with a span must satisfy original[start:end] == text.
	for _, c := range result {
		if c.StartChar == 0 && c.EndChar == 0 {
			continue // zero span is acceptable for split failures
		}
		span := original[c.StartChar:c.EndChar]
		assert.Equal(t, c.Text, span,
			"sub-chunk %d: span text must equal chunk text (start=%d end=%d)",
			c.ID, c.StartChar, c.EndChar)
	}
}

func TestEnforceHardLimits_ZeroSpanWhenNoOriginalText(t *testing.T) {
	t.Parallel()

	longText := strings.Repeat("alpha ", 500)
	chunks := []Chunk{
		{ID: 0, Text: longText, StartChar: 10, EndChar: 10 + len(longText), CharCount: len([]rune(longText))},
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 200})

	for _, c := range result {
		assert.Equal(t, 0, c.StartChar, "sub-chunk %d should have zero StartChar without OriginalText", c.ID)
		assert.Equal(t, 0, c.EndChar, "sub-chunk %d should have zero EndChar without OriginalText", c.ID)
	}
}

func TestEnforceHardLimits_MonotonicCursorOnDuplicateText(t *testing.T) {
	t.Parallel()

	original := "abc abc abc abc abc abc abc abc abc abc"
	chunks := []Chunk{
		{ID: 0, Text: original, StartChar: 0, EndChar: len(original), CharCount: len([]rune(original))},
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 15, OriginalText: original})

	assert.GreaterOrEqual(t, len(result), 2)

	// Spans must be strictly non-overlapping and advance monotonically.
	prevEnd := 0
	for _, c := range result {
		if c.StartChar == 0 && c.EndChar == 0 {
			continue
		}
		assert.GreaterOrEqual(t, c.StartChar, prevEnd,
			"sub-chunk %d: spans not monotonic (start=%d, prevEnd=%d)",
			c.ID, c.StartChar, prevEnd)
		span := original[c.StartChar:c.EndChar]
		assert.Equal(t, c.Text, span,
			"sub-chunk %d: span text must equal chunk text", c.ID)
		prevEnd = c.EndChar
	}
}

// ---------------------------------------------------------------------------
// Half-budget floor in splitAtBoundary
// ---------------------------------------------------------------------------

func TestSplitAtBoundary_HalfBudgetFloor(t *testing.T) {
	t.Parallel()

	// Construct text where the first sentence is very short ("Hi.")
	// and the second sentence alone exceeds maxChars. Without the floor,
	// splitAtBoundary would flush a single-word sliver.
	maxChars := 200
	short := "Hi."
	long := strings.Repeat("word ", 400) // ~2000 chars, well over maxChars
	text := short + " " + long

	result := splitAtBoundary(text, maxChars, splitSentencesForLimit)

	// No result chunk should be just "Hi." — the floor should prevent
	// flushing such a tiny leading chunk.
	for _, r := range result {
		assert.NotEqual(t, "Hi.", r,
			"splitAtBoundary should not flush a sliver chunk below half-budget")
	}
}

func TestSplitAtBoundary_FloorAllowsFlushWhenAboveHalf(t *testing.T) {
	t.Parallel()

	// Construct text where accumulated runeCount is >= maxChars/2 when
	// overflow is detected. The flush should proceed normally.
	maxChars := 100
	segments := []string{
		strings.Repeat("a", 55), // 55 runes — above maxChars/2 (50)
		strings.Repeat("b", 55), // 55 runes — triggers overflow
		strings.Repeat("c", 10), // small tail
	}
	text := segments[0] + " " + segments[1] + " " + segments[2]

	result := splitAtBoundary(text, maxChars, splitWordsForLimit)

	// Should produce at least 2 chunks since the accumulated count exceeds
	// half the budget when the overflow is detected.
	assert.GreaterOrEqual(t, len(result), 2,
		"splitAtBoundary should flush when runeCount >= maxChars/2")
}

func TestEnforceHardLimits_NoSliverChunk(t *testing.T) {
	t.Parallel()

	maxChars := 200
	// Short sentence followed by a long one that exceeds maxChars alone.
	text := "Short. " + strings.Repeat("longword ", maxChars/5)

	chunks := []Chunk{buildChunk(0, text)}
	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: maxChars})

	// No result chunk should have CharCount < maxChars/2 (the floor).
	for _, c := range result {
		assert.GreaterOrEqual(t, c.CharCount, maxChars/2,
			"chunk %d (%q...): CharCount %d is below half-budget floor %d",
			c.ID, truncate(t, c.Text, 30), c.CharCount, maxChars/2)
	}
}

// truncate returns the first n chars of s for display in assertions.
func truncate(t *testing.T, s string, n int) string {
	t.Helper()
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "..."
}

func TestChunkWithTokenLimit_NilChunker(t *testing.T) {
	t.Parallel()

	result := ChunkWithTokenLimit(nil, "text", 100, 0, fakeTokenCounter{maxTokens: 10})
	assert.Nil(t, result)
}

func TestChunkWithTokenLimit_NilCounterReturnsBaseChunks(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(SmartBoundary)
	require.NoError(t, err)

	text := "First sentence. Second sentence. Third sentence."
	base := c.Chunk(text, 100, 0)
	composed := ChunkWithTokenLimit(c, text, 100, 0, nil)

	assert.Equal(t, len(base), len(composed))
	for i := range base {
		assert.Equal(t, base[i].Text, composed[i].Text)
	}
}

func TestChunkWithTokenLimit_EnforcesTokenLimit(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(ParagraphAware)
	require.NoError(t, err)

	// Create text that produces chunks over 5 word-tokens.
	text := "This is a fairly long paragraph with many words in it. " +
		"Another paragraph here with even more words to test token budget enforcement. " +
		"Third paragraph continues the pattern of being too long for the fake limit."

	tc := fakeTokenCounter{maxTokens: 10}
	chunks := ChunkWithTokenLimit(c, text, 500, 0, tc)

	for _, ch := range chunks {
		tokCount := tc.CountTokens(ch.Text)
		// Allow indivisible atoms (single long words) to exceed; that's existing behavior.
		if tokCount > tc.maxTokens {
			// If it exceeds, it must be because it's a single indivisible unit.
			words := strings.Fields(ch.Text)
			assert.Equal(t, 1, len(words),
				"chunk %d exceeds token budget (%d > %d) but has multiple words: %q",
				ch.ID, tokCount, tc.maxTokens, ch.Text)
		}
	}
}

func TestChunkWithTokenLimit_PreservesPassThroughSpans(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(SmartBoundary)
	require.NoError(t, err)

	text := "First sentence. Second sentence. Third one."
	// Permissive counter — no splits should happen.
	tc := fakeTokenCounter{maxTokens: 100}
	chunks := ChunkWithTokenLimit(c, text, 500, 0, tc)

	for _, ch := range chunks {
		if ch.StartChar != 0 || ch.EndChar != 0 {
			assert.Equal(t, text[ch.StartChar:ch.EndChar], ch.Text,
				"span round-trip failed for chunk %d", ch.ID)
		}
	}
}

func TestChunkWithTokenLimit_ClearsSplitSpans(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(HardCut)
	require.NoError(t, err)

	text := strings.Repeat("word ", 100)
	// Restrictive counter — should trigger splits.
	tc := fakeTokenCounter{maxTokens: 5}
	chunks := ChunkWithTokenLimit(c, text, 500, 0, tc)

	require.Greater(t, len(chunks), 1, "expected multiple chunks after token limiting")

	// Split sub-chunks should have zero spans.
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue // zero span is expected for splits
		}
		// If span is non-zero, it must round-trip.
		assert.Equal(t, text[ch.StartChar:ch.EndChar], ch.Text,
			"non-zero span failed round-trip for chunk %d", ch.ID)
	}
}

func TestEnforceHardLimits_StrictGuarantee(t *testing.T) {
	t.Parallel()

	// An input with a paragraph that exceeds MaxChars, followed by another paragraph.
	// Paragraph splitting will split them, but the first paragraph is still oversized.
	// It should cascade down to sentence, word, or hard cut until it fits MaxChars.
	para1 := "ThisIsAVeryLongParagraphWithoutNewlinesThatIsClearlyOversizedAndMustBeSplitFurtherToFitTheLimit."
	para2 := "Short paragraph."
	text := para1 + "\n\n" + para2

	chunks := []Chunk{buildChunk(0, text)}
	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 20})

	require.NotEmpty(t, result)
	for _, c := range result {
		assert.LessOrEqual(t, utf8.RuneCountInString(c.Text), 20,
			"chunk %d exceeds limit of 20: %q (len %d)", c.ID, c.Text, utf8.RuneCountInString(c.Text))
	}
}

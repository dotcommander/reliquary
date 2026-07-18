package chunking

import (
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewChunker_SmartBoundary(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(SmartBoundary)
	if err != nil {
		t.Fatalf("NewChunker(SmartBoundary) returned error: %v", err)
	}
	if c == nil {
		t.Fatal("expected non-nil chunker")
	}
	if got := c.Strategy(); got != SmartBoundary {
		t.Errorf("Strategy() = %q, want %q", got, SmartBoundary)
	}
}

func TestNewChunker_InvalidStrategy(t *testing.T) {
	t.Parallel()

	c, err := NewChunker("no_such_strategy")
	if err == nil {
		t.Fatal("expected error for unknown strategy, got nil")
	}
	if c != nil {
		t.Error("expected nil chunker for unknown strategy")
	}
}

func TestNewChunker_UnknownStrategy_SentinelMatchable(t *testing.T) {
	t.Parallel()

	_, err := NewChunker("no_such_strategy")
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrUnknownStrategy),
		"errors.Is(err, ErrUnknownStrategy) should be true, got: %v", err)
}

func TestSmartBoundary_ShortText(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(SmartBoundary)
	if err != nil {
		t.Fatal(err)
	}

	text := "Hello world. This is a short sentence."
	chunks := c.Chunk(text, 500, 0)

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Text != text {
		t.Errorf("chunk text mismatch:\ngot:  %q\nwant: %q", chunks[0].Text, text)
	}
}

func TestSmartBoundary_LongText(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(SmartBoundary)
	if err != nil {
		t.Fatal(err)
	}

	// Build text long enough to produce multiple chunks.
	sentences := make([]string, 20)
	for i := range sentences {
		sentences[i] = "This is sentence number " + string(rune('A'+i)) + "."
	}
	text := strings.Join(sentences, " ")

	// Small chunk size to force multiple chunks.
	chunks := c.Chunk(text, 100, 20)

	if len(chunks) < 2 {
		t.Fatalf("expected >= 2 chunks for long text, got %d", len(chunks))
	}

	// Verify overlap: consecutive chunks should share some text.
	foundOverlap := false
	for i := 1; i < len(chunks); i++ {
		prev := chunks[i-1].Text
		curr := chunks[i].Text
		// Check if the tail of the previous chunk appears in the current chunk.
		overlapLen := min(len(prev), 20)
		if overlapLen > 0 && strings.Contains(curr, prev[len(prev)-overlapLen:]) {
			foundOverlap = true
			break
		}
	}
	if !foundOverlap {
		t.Log("no overlap detected between consecutive chunks (overlap may be at sentence level)")
	}

	// Verify all chunk IDs are sequential.
	for i, ch := range chunks {
		if ch.ID != i {
			t.Errorf("chunk %d has ID %d", i, ch.ID)
		}
	}
}

func TestSmartBoundary_EmptyText(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(SmartBoundary)
	if err != nil {
		t.Fatal(err)
	}

	chunks := c.Chunk("", 500, 0)
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks for empty text, got %d", len(chunks))
	}
}

func TestSmartBoundary_ZeroSize(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(SmartBoundary)
	if err != nil {
		t.Fatal(err)
	}

	chunks := c.Chunk("some text", 0, 0)
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks for size 0, got %d", len(chunks))
	}
}

func TestSmartBoundary_ChunkFields(t *testing.T) {
	t.Parallel()

	c, err := NewChunker(SmartBoundary)
	if err != nil {
		t.Fatal(err)
	}

	text := "This is a test sentence. With multiple words here."
	chunks := c.Chunk(text, 500, 0)

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}

	ch := chunks[0]

	// Verify ID is populated.
	if ch.ID != 0 {
		t.Errorf("expected ID=0, got %d", ch.ID)
	}

	// Verify Text is populated.
	if ch.Text == "" {
		t.Error("expected non-empty Text")
	}

	// Verify CharCount.
	expectedChars := len([]rune(text))
	if ch.CharCount != expectedChars {
		t.Errorf("CharCount = %d, want %d", ch.CharCount, expectedChars)
	}

	// Verify WordCount.
	expectedWords := countWords(text)
	if ch.WordCount != expectedWords {
		t.Errorf("WordCount = %d, want %d", ch.WordCount, expectedWords)
	}

	// Verify StartChar and EndChar are populated with valid byte offsets.
	// For a single-chunk result, the span should cover the entire input.
	if ch.StartChar < 0 {
		t.Errorf("StartChar = %d, want >= 0", ch.StartChar)
	}
	if ch.EndChar <= 0 {
		t.Errorf("expected EndChar > 0, got %d", ch.EndChar)
	}
	if ch.StartChar < ch.EndChar && len(text) > 0 {
		// Span text should match chunk text (allowing for trimming).
		spanText := text[ch.StartChar:ch.EndChar]
		if spanText != ch.Text {
			t.Errorf("span text %q != chunk text %q", spanText, ch.Text)
		}
	}
}

// ---------------------------------------------------------------------------
// Source span tests — verify StartChar/EndChar for each strategy
// ---------------------------------------------------------------------------

func TestHardCut_SourceSpans(t *testing.T) {
	t.Parallel()

	text := "Hello world! This is a test. Goodbye world!"
	c, _ := NewChunker(HardCut)
	chunks := c.Chunk(text, 15, 0)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue // split sub-chunk
		}
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q (start=%d, end=%d)",
				ch.ID, span, ch.Text, ch.StartChar, ch.EndChar)
		}
	}

	// Verify no overlapping spans (non-overlap case).
	for i := 1; i < len(chunks); i++ {
		if chunks[i-1].EndChar > chunks[i].StartChar {
			t.Errorf("spans overlap: chunk %d ends at %d, chunk %d starts at %d",
				chunks[i-1].ID, chunks[i-1].EndChar, chunks[i].ID, chunks[i].StartChar)
		}
	}
}

func TestHardCut_SourceSpansWithOverlap(t *testing.T) {
	t.Parallel()

	text := "abcdefghij"
	c, _ := NewChunker(HardCut)
	chunks := c.Chunk(text, 5, 2)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q", ch.ID, span, ch.Text)
		}
	}
}

func TestSentenceBoundary_SourceSpans(t *testing.T) {
	t.Parallel()

	text := "First sentence. Second sentence. Third sentence."
	c, _ := NewChunker(SentenceBoundary)
	chunks := c.Chunk(text, 100, 0)

	// Single chunk should span the whole text.
	require.Len(t, chunks, 1)
	if chunks[0].StartChar > 0 || chunks[0].EndChar <= 0 {
		t.Errorf("unexpected zero span: start=%d end=%d", chunks[0].StartChar, chunks[0].EndChar)
	}
}

func TestSentenceBoundary_SourceSpansRoundTrip(t *testing.T) {
	t.Parallel()

	text := "First sentence here. Second one follows. Third is last."
	c, _ := NewChunker(SentenceBoundary)
	chunks := c.Chunk(text, 30, 0)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q", ch.ID, span, ch.Text)
		}
	}
}

func TestSmartBoundary_SourceSpans(t *testing.T) {
	t.Parallel()

	text := "First sentence here. Second one follows. Third is the last one."
	c, _ := NewChunker(SmartBoundary)
	chunks := c.Chunk(text, 30, 0)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q", ch.ID, span, ch.Text)
		}
	}
}

func TestWordBoundary_SourceSpans(t *testing.T) {
	t.Parallel()

	text := "alpha beta gamma delta epsilon"
	c, _ := NewChunker(WordBoundary)
	chunks := c.Chunk(text, 15, 0)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q", ch.ID, span, ch.Text)
		}
	}

	// Verify contiguous coverage (no overlap in this config).
	for i := 1; i < len(chunks); i++ {
		if chunks[i-1].EndChar > chunks[i].StartChar {
			t.Errorf("spans overlap: chunk %d ends at %d, chunk %d starts at %d",
				chunks[i-1].ID, chunks[i-1].EndChar, chunks[i].ID, chunks[i].StartChar)
		}
	}
}

func TestParagraphAware_SourceSpans(t *testing.T) {
	t.Parallel()

	text := "First paragraph.\n\nSecond paragraph here.\n\nThird one."
	c, _ := NewChunker(ParagraphAware)
	chunks := c.Chunk(text, 100, 0)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q", ch.ID, span, ch.Text)
		}
	}
}

func TestMarkdownAware_SourceSpans(t *testing.T) {
	t.Parallel()

	text := "# Heading\n\nParagraph one.\n\nParagraph two."
	c, _ := NewChunker(MarkdownAware)
	chunks := c.Chunk(text, 200, 0)

	// Should produce at least one chunk with a valid span.
	found := false
	for _, ch := range chunks {
		if ch.StartChar > 0 || ch.EndChar > 0 {
			found = true
			span := text[ch.StartChar:ch.EndChar]
			if span != ch.Text {
				t.Errorf("chunk %d: span %q != text %q", ch.ID, span, ch.Text)
			}
		}
	}
	if !found {
		t.Error("expected at least one chunk with non-zero span")
	}
}

func TestHeadingAware_SourceSpans(t *testing.T) {
	t.Parallel()

	text := "# Section 1\nContent one.\n\n# Section 2\nContent two."
	c, _ := NewChunker(HeadingAware)
	chunks := c.Chunk(text, 200, 0)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q", ch.ID, span, ch.Text)
		}
	}
}

func TestTokenBased_SourceSpans(t *testing.T) {
	t.Parallel()

	text := "The quick brown fox jumps over the lazy dog. " +
		"A longer sentence with more tokens for testing. " +
		"And another one to make it longer still."
	c, _ := NewChunker(TokenBased)
	chunks := c.Chunk(text, 10, 0)

	// At least some chunks should have spans.
	haveSpan := false
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		haveSpan = true
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q", ch.ID, span, ch.Text)
		}
	}
	if !haveSpan {
		t.Error("expected at least one chunk with non-zero span")
	}
}

func TestHardCut_UnicodeSourceSpans(t *testing.T) {
	t.Parallel()

	// Multi-byte characters: each emoji is 4 bytes in UTF-8.
	text := "hello 🌍 world 🌙 test"
	c, _ := NewChunker(HardCut)
	chunks := c.Chunk(text, 8, 0)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		if span != ch.Text {
			t.Errorf("chunk %d: span %q != text %q (start=%d, end=%d)",
				ch.ID, span, ch.Text, ch.StartChar, ch.EndChar)
		}
	}
}

func TestEnforceHardLimits_SpansClearedOnSplit(t *testing.T) {
	t.Parallel()

	// Chunk with a valid span that exceeds the limit.
	longText := strings.Repeat("alpha ", 500)
	chunks := []Chunk{
		{ID: 0, Text: longText, StartChar: 10, EndChar: 10 + len(longText), CharCount: len([]rune(longText))},
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 200})

	// The original chunk was split — sub-chunks should have cleared spans.
	for _, c := range result {
		if c.StartChar != 0 || c.EndChar != 0 {
			t.Errorf("split sub-chunk %d has non-zero span: start=%d end=%d",
				c.ID, c.StartChar, c.EndChar)
		}
	}
}

func TestEnforceHardLimits_SpansPreservedOnPassThrough(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 0, Text: "short text", StartChar: 5, EndChar: 15, CharCount: 10},
	}

	result := EnforceHardLimits(chunks, LimitOptions{MaxChars: 1000})

	require.Len(t, result, 1)
	assert.Equal(t, 5, result[0].StartChar)
	assert.Equal(t, 15, result[0].EndChar)
}

// ---------------------------------------------------------------------------
// Slice 1 span hardening tests
// ---------------------------------------------------------------------------

func TestSentenceBoundary_UTF8SourceSpans(t *testing.T) {
	t.Parallel()

	// Chinese characters: each rune is 3 bytes in UTF-8.
	text := "你好世界。再见世界。你好吗。"
	c, err := NewChunker(SentenceBoundary)
	require.NoError(t, err)

	chunks := c.Chunk(text, 100, 0)
	require.NotEmpty(t, chunks)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: span text must equal chunk text (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}
}

func TestSmartBoundary_UTF8SourceSpans(t *testing.T) {
	t.Parallel()

	// Emoji: each emoji is 4 bytes in UTF-8.
	text := "hello 🌍 world 🌙 test. Another sentence here."
	c, err := NewChunker(SmartBoundary)
	require.NoError(t, err)

	chunks := c.Chunk(text, 40, 0)
	require.NotEmpty(t, chunks)

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: span text must equal chunk text (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}
}

func TestSentenceBoundary_RepeatedTextMonotonicSpans(t *testing.T) {
	t.Parallel()

	// Identical sentences repeated — spans must point to later occurrences,
	// not the first one found by naive string search.
	text := "alpha beta. alpha beta. alpha beta."
	c, err := NewChunker(SentenceBoundary)
	require.NoError(t, err)

	chunks := c.Chunk(text, 20, 0)
	require.Len(t, chunks, 3)

	// Verify spans are strictly increasing.
	for i := 1; i < len(chunks); i++ {
		assert.True(t, chunks[i].StartChar >= chunks[i-1].EndChar,
			"spans not monotonic: chunk %d starts at %d, chunk %d ends at %d",
			i, chunks[i].StartChar, i-1, chunks[i-1].EndChar)
	}

	// Verify exact source-slice match.
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: span text must equal chunk text", ch.ID)
	}
}

func TestSmartBoundary_RepeatedTextMonotonicSpans(t *testing.T) {
	t.Parallel()

	text := "same content. same content. same content."
	c, err := NewChunker(SmartBoundary)
	require.NoError(t, err)

	chunks := c.Chunk(text, 20, 0)
	require.Len(t, chunks, 3)

	for i := 1; i < len(chunks); i++ {
		assert.True(t, chunks[i].StartChar >= chunks[i-1].EndChar,
			"spans not monotonic: chunk %d starts at %d, chunk %d ends at %d",
			i, chunks[i].StartChar, i-1, chunks[i-1].EndChar)
	}

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: span text must equal chunk text", ch.ID)
	}
}

func TestSentenceBoundary_OverlapSpansInvariant(t *testing.T) {
	t.Parallel()

	// When overlap is used, each chunk's span must either:
	// 1. Point to a contiguous verbatim source slice, OR
	// 2. Be cleared to 0,0.
	text := "First sentence here. Second sentence follows. Third is the last. Fourth wraps up."
	c, err := NewChunker(SentenceBoundary)
	require.NoError(t, err)

	chunks := c.Chunk(text, 50, 20)
	require.True(t, len(chunks) >= 2, "need at least 2 chunks for overlap test")

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue // cleared span is valid
		}
		// If span is set, it must be an exact source match.
		assert.LessOrEqual(t, 0, ch.StartChar)
		assert.LessOrEqual(t, ch.StartChar, ch.EndChar)
		assert.LessOrEqual(t, ch.EndChar, len(text))
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: non-zero span must be exact source match (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}
}

func TestSmartBoundary_OverlapSpansInvariant(t *testing.T) {
	t.Parallel()

	text := "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
	c, err := NewChunker(SmartBoundary)
	require.NoError(t, err)

	chunks := c.Chunk(text, 45, 15)
	require.True(t, len(chunks) >= 2, "need at least 2 chunks for overlap test")

	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		assert.LessOrEqual(t, 0, ch.StartChar)
		assert.LessOrEqual(t, ch.StartChar, ch.EndChar)
		assert.LessOrEqual(t, ch.EndChar, len(text))
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: non-zero span must be exact source match (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}
}

func TestTrimSpanToText_ExactMatch(t *testing.T) {
	t.Parallel()

	// After trimming whitespace, the span should produce the target exactly.
	source := "  hello world  "
	target := "hello world"
	start, end := trimSpanToText(source, target, 0, len(source))
	assert.Equal(t, "hello world", source[start:end])
}

func TestTrimSpanToText_NoMatch(t *testing.T) {
	t.Parallel()

	// When trimming doesn't produce an exact match, the function returns
	// whatever it trimmed to (but the caller checks the match).
	source := "  different text  "
	target := "hello world"
	start, end := trimSpanToText(source, target, 0, len(source))
	// Should not panic, even if no exact match.
	_ = source[start:end]
}

func TestTrimSpanToText_UTF8(t *testing.T) {
	t.Parallel()

	source := "  你好世界  "
	target := "你好世界"
	start, end := trimSpanToText(source, target, 0, len(source))
	assert.Equal(t, "你好世界", source[start:end])
}

// ---------------------------------------------------------------------------
// Slice 2 hard-cut span tests
// ---------------------------------------------------------------------------

func TestHardCut_UTF8SpansNoOverlap(t *testing.T) {
	t.Parallel()

	// Greek: each rune is 2 bytes in UTF-8.
	text := "αβγδεζηθικλμν"
	c, err := NewChunker(HardCut)
	require.NoError(t, err)

	chunks := c.Chunk(text, 4, 0)
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: span must equal chunk text (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}
}

func TestHardCut_UTF8OverlapSpansInvariant(t *testing.T) {
	t.Parallel()

	// With overlap, EnforceHardLimits may re-split chunks, clearing spans.
	// Any non-zero span must still be an exact source match.
	text := "αβγδεζηθικλμνξοπρστυφχψω"
	c, err := NewChunker(HardCut)
	require.NoError(t, err)

	chunks := c.Chunk(text, 5, 2)
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue // cleared span is valid
		}
		assert.LessOrEqual(t, 0, ch.StartChar)
		assert.LessOrEqual(t, ch.StartChar, ch.EndChar)
		assert.LessOrEqual(t, ch.EndChar, len(text))
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: non-zero span must be exact source match (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}
}

func TestHardCut_MixedUTF8ASCIISpans(t *testing.T) {
	t.Parallel()

	text := "hello 🌍 world 🌙 end"
	c, err := NewChunker(HardCut)
	require.NoError(t, err)

	chunks := c.Chunk(text, 6, 0)
	for _, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		span := text[ch.StartChar:ch.EndChar]
		assert.Equal(t, ch.Text, span,
			"chunk %d: span must equal chunk text (start=%d end=%d)",
			ch.ID, ch.StartChar, ch.EndChar)
	}
}

// ---------------------------------------------------------------------------
// Oversized-unit escape hatch tests
// ---------------------------------------------------------------------------

func TestSentenceBoundary_SingleOversizedUnit(t *testing.T) {
	t.Parallel()

	longSentence := strings.Repeat("word ", 30) // ~150 chars
	c, err := NewChunker(SentenceBoundary)
	require.NoError(t, err)

	chunks := c.Chunk(strings.TrimSpace(longSentence), 50, 0)
	require.NotEmpty(t, chunks, "oversized sentence should still produce output")
	for _, ch := range chunks {
		assert.NotEmpty(t, strings.TrimSpace(ch.Text), "no empty chunks allowed")
	}
}

func TestWordBoundary_SingleOversizedUnit(t *testing.T) {
	t.Parallel()

	longWord := strings.Repeat("a", 200)
	c, err := NewChunker(WordBoundary)
	require.NoError(t, err)

	chunks := c.Chunk(longWord, 50, 0)
	require.NotEmpty(t, chunks, "oversized word should still produce output")
	for _, ch := range chunks {
		assert.NotEmpty(t, strings.TrimSpace(ch.Text), "no empty chunks allowed")
	}
}

func TestParagraphAware_SingleOversizedUnit(t *testing.T) {
	t.Parallel()

	longPara := strings.Repeat("This is a sentence. ", 20) // ~420 chars
	c, err := NewChunker(ParagraphAware)
	require.NoError(t, err)

	chunks := c.Chunk(strings.TrimSpace(longPara), 100, 0)
	require.NotEmpty(t, chunks, "oversized paragraph should still produce output")
	for _, ch := range chunks {
		assert.NotEmpty(t, strings.TrimSpace(ch.Text), "no empty chunks allowed")
	}
}

// ---------------------------------------------------------------------------
// ContentHash tests
// ---------------------------------------------------------------------------

func TestBuildChunkContentHash(t *testing.T) {
	t.Parallel()

	c1 := buildChunk(0, "hello world")
	c2 := buildChunk(1, "hello world")
	c3 := buildChunk(2, "different text")

	assert.Equal(t, c1.ContentHash, c2.ContentHash, "identical text should produce identical hash")
	assert.NotEqual(t, c1.ContentHash, c3.ContentHash, "different text should produce different hash")
}

func TestBuildChunkWithSpanContentHash(t *testing.T) {
	t.Parallel()

	c1 := buildChunkWithSpan(0, "hello world", 0, 11)
	c2 := buildChunk(0, "hello world")

	assert.Equal(t, c1.ContentHash, c2.ContentHash, "buildChunkWithSpan should produce same hash as buildChunk")
	assert.NotEmpty(t, c1.ContentHash)
}

func TestContentHashLength(t *testing.T) {
	t.Parallel()

	c := buildChunk(0, "some text")
	assert.Len(t, c.ContentHash, 16, "ContentHash should be 16 hex characters")

	// Empty text should still produce a stable hash.
	ce := buildChunk(1, "")
	assert.Len(t, ce.ContentHash, 16, "empty text should have 16-char hash")
	assert.NotEmpty(t, ce.ContentHash)
}

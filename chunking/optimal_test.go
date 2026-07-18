package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// OptimalChunker SplitIntoChunks behavior
// ---------------------------------------------------------------------------

func TestSplitIntoChunks_SingleChunkForShortInput(t *testing.T) {
	t.Parallel()

	c := NewOptimalChunker()
	text := strings.Repeat("x", 500) // well below OptimalLength (10000)

	chunks := c.SplitIntoChunks(text)
	require.Len(t, chunks, 1)
	assert.Equal(t, text, chunks[0])
}

func TestSplitIntoChunks_BoundaryPreference_ParagraphBreak(t *testing.T) {
	t.Parallel()

	c := &OptimalChunker{OptimalLength: 200, MinLength: 50, MaxLength: 300}

	// Build text with a paragraph break in the upper half of the window.
	// Total must exceed OptimalLength so a split is needed.
	head := strings.Repeat("a", 120) // > OptimalLength/2 (100)
	tail := strings.Repeat("b", 200) // pushes total to 322 > 200
	text := head + "\n\n" + tail

	chunks := c.SplitIntoChunks(text)

	require.GreaterOrEqual(t, len(chunks), 2, "should split at paragraph break")
	// First chunk should end with the paragraph break.
	assert.Contains(t, chunks[0], "\n\n", "first chunk should contain the paragraph break")
	assert.True(t, strings.HasPrefix(chunks[0], head), "first chunk should start with head")
}

func TestSplitIntoChunks_BoundaryPreference_SentenceBreak(t *testing.T) {
	t.Parallel()

	c := &OptimalChunker{OptimalLength: 200, MinLength: 50, MaxLength: 300}

	// Build text with ". " in upper half, no "\n\n".
	// Total must exceed OptimalLength.
	head := strings.Repeat("a", 120) // > OptimalLength/2
	tail := strings.Repeat("b", 200) // pushes total over 200
	text := head + ". " + tail

	chunks := c.SplitIntoChunks(text)

	require.GreaterOrEqual(t, len(chunks), 2, "should split at sentence break")
	// First chunk should end with ". "
	assert.True(t, strings.HasSuffix(chunks[0], ". "), "first chunk should end at sentence break")
}

func TestSplitIntoChunks_HardCutWhenNoBoundaryInUpperHalf(t *testing.T) {
	t.Parallel()

	c := &OptimalChunker{OptimalLength: 200, MinLength: 50, MaxLength: 300}

	// Build text with ". " only in the lower half (< OptimalLength/2).
	head := strings.Repeat("a", 50) // < OptimalLength/2 (100)
	head += ". "
	head += strings.Repeat("b", 50)
	tail := strings.Repeat("c", 300)
	text := head + tail

	chunks := c.SplitIntoChunks(text)

	// Should hard-cut at OptimalLength, not at the early ". "
	require.GreaterOrEqual(t, len(chunks), 2, "should split into multiple chunks")
	// First chunk should be exactly OptimalLength (hard cut).
	assert.Equal(t, c.OptimalLength, len(chunks[0]),
		"first chunk should hard-cut at OptimalLength when no boundary in upper half")
}

func TestSplitIntoChunks_ReconstructsContent(t *testing.T) {
	t.Parallel()

	c := &OptimalChunker{OptimalLength: 200, MinLength: 50, MaxLength: 300}

	// Paragraph-break-aligned content should reconstruct exactly.
	segments := make([]string, 20)
	for i := range segments {
		segments[i] = strings.Repeat("seg", 30)
	}
	text := strings.Join(segments, "\n\n")

	chunks := c.SplitIntoChunks(text)
	reconstructed := strings.Join(chunks, "")
	assert.Equal(t, text, reconstructed,
		"SplitIntoChunks with paragraph breaks should reconstruct content losslessly")
}

// ---------------------------------------------------------------------------
// OptimalChunker Chunk interface
// ---------------------------------------------------------------------------

func TestOptimalChunker_ChunkInterface_EmptyInput(t *testing.T) {
	t.Parallel()

	c := NewOptimalChunker()
	result := c.Chunk("", 1000, 0)
	assert.Nil(t, result, "empty text should return nil")
}

func TestOptimalChunker_ChunkInterface_ChunksReconstruct(t *testing.T) {
	t.Parallel()

	c := NewOptimalChunker()
	text := strings.Repeat("This is a test sentence. ", 500) // ~12.5k chars

	chunks := c.Chunk(text, 1000, 0)
	require.NotEmpty(t, chunks, "should produce chunks for long text")

	// Reconstruct from chunk texts.
	reconstructed := strings.Join(func() []string {
		texts := make([]string, len(chunks))
		for i, ch := range chunks {
			texts[i] = ch.Text
		}
		return texts
	}(), "")

	assert.Equal(t, text, reconstructed, "chunks should reconstruct original text")
}

// ---------------------------------------------------------------------------
// NewChunker registration
// ---------------------------------------------------------------------------

func TestNewChunker_OptimalStrategy(t *testing.T) {
	t.Parallel()

	chunker, err := NewChunker(Optimal)
	require.NoError(t, err)
	require.NotNil(t, chunker)
	assert.Equal(t, Optimal, chunker.Strategy())

	// Verify it's an *OptimalChunker.
	_, ok := chunker.(*OptimalChunker)
	assert.True(t, ok, "NewChunker(Optimal) should return *OptimalChunker")
}

// ---------------------------------------------------------------------------
// ChunkContent pass-through tiers
// ---------------------------------------------------------------------------

func TestOptimalChunker_ShortPassThrough(t *testing.T) {
	t.Parallel()

	c := NewOptimalChunker()
	text := "This is short content."
	result := c.ChunkContent(text)
	assert.Equal(t, text, result, "content below MinLength should pass through unchanged")
}

func TestOptimalChunker_OptimalRangePassThrough(t *testing.T) {
	t.Parallel()

	c := NewOptimalChunker()
	// Between MinLength (5000) and MaxLength (15000).
	text := strings.Repeat("x", 10000)
	result := c.ChunkContent(text)
	assert.Equal(t, text, result, "content in optimal range should pass through unchanged")
}

// ---------------------------------------------------------------------------
// StrategicSample behavior
// ---------------------------------------------------------------------------

func TestStrategicSample_BelowBudget(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("a", 500)
	result := StrategicSample(text, 1000)
	assert.Equal(t, text, result, "text below optimalLen should return unchanged")
}

func TestStrategicSample_ZeroLen(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("a", 500)
	result := StrategicSample(text, 0)
	assert.Equal(t, text, result, "optimalLen 0 should return text unchanged")

	result = StrategicSample(text, -5)
	assert.Equal(t, text, result, "negative optimalLen should return text unchanged")
}

func TestStrategicSample_SeamPositions(t *testing.T) {
	t.Parallel()

	// optimalLen=600: firstPortion=400, remainingQuota=200, sampleSize=66
	text := strings.Repeat("a", 1000)
	result := StrategicSample(text, 600)

	assert.Contains(t, result, seamMiddle, "should contain middle seam marker")
	assert.Contains(t, result, seamEnd, "should contain end seam marker")

	// Result should start with the first 400 bytes.
	assert.True(t, strings.HasPrefix(result, text[:400]),
		"should start with firstPortion")

	// Result should be shorter than original.
	assert.Less(t, len(result), len(text), "result should be shorter than original")
}

func TestStrategicSample_NoTrailingMarker(t *testing.T) {
	t.Parallel()

	text := strings.Repeat("x", 20000)
	result := StrategicSample(text, 10000)

	assert.NotContains(t, result, "Content optimized for AI processing",
		"should not contain the omitted mr trailing marker")
}

func TestStrategicSample_OverlapGuard(t *testing.T) {
	t.Parallel()

	// Construct input where endPoint would overlap with middle sample.
	// With optimalLen=10, firstPortion=6, remainingQuota=4, sampleSize=1
	// midPoint of remaining=4/2=2, endPoint=4-1=3
	// midPoint+sampleSize=3, endPoint=3 → no overlap yet, but close.
	// With optimalLen=8, firstPortion=5, remainingQuota=3, sampleSize=1
	// remaining=3 chars, midPoint=1, endPoint=2, midPoint+sampleSize=2, endPoint=2 → overlap!
	text := "abcdefgh" // 8 bytes
	result := StrategicSample(text, 5)

	// Must not panic and must produce non-empty output.
	assert.NotEmpty(t, result, "overlap guard should produce output")
}

func TestOptimalChunker_StrategicSampleReceiver(t *testing.T) {
	t.Parallel()

	c := NewOptimalChunker()
	text := strings.Repeat("x", 20000)

	receiverResult := c.StrategicSample(text)
	packageResult := StrategicSample(text, 10000) // c.OptimalLength

	assert.Equal(t, packageResult, receiverResult,
		"receiver should delegate to package function with c.OptimalLength")
}

func TestOptimalChunker_OversizedStrategicSample(t *testing.T) {
	t.Parallel()

	c := NewOptimalChunker()
	text := strings.Repeat("x", 20000) // well above MaxLength (15000)

	result := c.ChunkContent(text)
	assert.Contains(t, result, seamMiddle,
		"oversized content should contain middle seam marker")
	assert.Contains(t, result, seamEnd,
		"oversized content should contain end seam marker")
	assert.Less(t, len(result), len(text),
		"oversized content should be shorter than original")
}

// ---------------------------------------------------------------------------
// OptimalChunker span contract
// ---------------------------------------------------------------------------

func TestOptimalChunker_ChunkInterface_SpanRoundTrip(t *testing.T) {
	t.Parallel()

	c := &OptimalChunker{OptimalLength: 200, MinLength: 50, MaxLength: 300}
	segments := make([]string, 10)
	for i := range segments {
		segments[i] = "This is paragraph " + strings.Repeat("x", 50) + "."
	}
	text := strings.Join(segments, "\n\n")

	chunks := c.Chunk(text, 200, 0)
	require.NotEmpty(t, chunks)

	for i, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue // zero span = unknown
		}
		assert.Equal(t, ch.Text, text[ch.StartChar:ch.EndChar],
			"chunk %d: text[start:end] must equal chunk text", i)
		assert.True(t, ch.EndChar > ch.StartChar,
			"chunk %d: EndChar must be > StartChar", i)
	}
}

func TestOptimalChunker_ChunkInterface_UnicodeSpanRoundTrip(t *testing.T) {
	t.Parallel()

	c := &OptimalChunker{OptimalLength: 100, MinLength: 20, MaxLength: 150}
	// Unicode-heavy text with multi-byte runes.
	text := strings.Repeat("Café naïve résumé Über 漢字 ひらがな 한글 🚀 ", 20)

	chunks := c.Chunk(text, 100, 0)
	require.NotEmpty(t, chunks)

	reconstructed := strings.Join(func() []string {
		texts := make([]string, len(chunks))
		for i, ch := range chunks {
			texts[i] = ch.Text
		}
		return texts
	}(), "")

	assert.Equal(t, text, reconstructed, "chunks must reconstruct original text")

	for i, ch := range chunks {
		if ch.StartChar == 0 && ch.EndChar == 0 {
			continue
		}
		assert.Equal(t, ch.Text, text[ch.StartChar:ch.EndChar],
			"chunk %d: Unicode span round-trip failed", i)
		// Verify all chunk text is valid UTF-8.
		assert.True(t, utf8.ValidString(ch.Text),
			"chunk %d: text must be valid UTF-8", i)
	}
}

func TestOptimalChunker_ChunkInterface_RepeatedTextSpansMonotonic(t *testing.T) {
	t.Parallel()

	c := &OptimalChunker{OptimalLength: 100, MinLength: 20, MaxLength: 150}
	// Repeated identical paragraphs — spans must advance, not all point at first occurrence.
	para := "Repeated content for testing monotonic spans. "
	text := strings.Repeat(para, 50) // 50 × ~48 bytes

	chunks := c.Chunk(text, 100, 0)
	require.GreaterOrEqual(t, len(chunks), 2, "need multiple chunks for monotonicity check")

	for i := 1; i < len(chunks); i++ {
		prev := chunks[i-1]
		cur := chunks[i]
		if prev.StartChar == 0 && prev.EndChar == 0 {
			continue
		}
		if cur.StartChar == 0 && cur.EndChar == 0 {
			continue
		}
		assert.GreaterOrEqual(t, cur.StartChar, prev.EndChar,
			"chunk %d: StartChar (%d) must be >= prev EndChar (%d) — spans must be monotonic",
			i, cur.StartChar, prev.EndChar)
	}
}

func TestSplitIntoChunks_UnicodeDoesNotSplitCodePoint(t *testing.T) {
	t.Parallel()

	c := &OptimalChunker{OptimalLength: 10, MinLength: 5, MaxLength: 15}
	// Each emoji is 4 bytes, each CJK char is 3 bytes. Size=10 runes.
	text := strings.Repeat("🚀星🌟月✨", 20) // 5 runes × 20 = 100 runes, ~300 bytes

	chunks := c.SplitIntoChunks(text)
	require.GreaterOrEqual(t, len(chunks), 2)

	for i, chunk := range chunks {
		assert.True(t, utf8.ValidString(chunk),
			"chunk %d: must be valid UTF-8, not split mid-codepoint", i)
	}

	reconstructed := strings.Join(chunks, "")
	assert.Equal(t, text, reconstructed, "reconstruction must be lossless")
}

func TestOptimalChunker_ChunkInterface_OverlapIgnoredDocumented(t *testing.T) {
	t.Parallel()

	c := NewOptimalChunker()
	text := strings.Repeat("Overlap test content. ", 500)

	// overlap=100 should have no effect — documented behavior.
	noOverlap := c.Chunk(text, 1000, 0)
	withOverlap := c.Chunk(text, 1000, 100)

	require.Equal(t, len(noOverlap), len(withOverlap),
		"overlap parameter should not affect OptimalChunker output")

	for i := range noOverlap {
		assert.Equal(t, noOverlap[i].Text, withOverlap[i].Text,
			"chunk %d: overlap should not change chunk text", i)
	}
}

func TestStrategicSample_UnicodeRuneByteSafety(t *testing.T) {
	t.Parallel()

	// 🚀 is 4 bytes, 漢字 is 6 bytes (3 each).
	// Let's create a string of 10000 runes (which will be much larger in bytes).
	var sb strings.Builder
	for range 2000 {
		sb.WriteString("🚀Café漢字")
	}
	text := sb.String() // 14000 runes, 26000 bytes

	// optimalLen is 10000 runes.
	// Since 14000 > 10000, it should sample it.
	sampled := StrategicSample(text, 10000)

	assert.True(t, utf8.ValidString(sampled), "sampled text must be valid UTF-8")
	// The result should not be the original text since it was sampled.
	assert.NotEqual(t, text, sampled)
}

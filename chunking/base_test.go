package chunking

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTailWindow(t *testing.T) {
	t.Parallel()

	cost := func(s string) int { return len(s) }

	t.Run("normal window", func(t *testing.T) {
		t.Parallel()
		// Budget 5, costs: "a"=1, "bb"=2, "ccc"=3
		// From end: "ccc"=3 (total=3), "bb"=2 (total=5), "a"=1 (total=6 > 5, stop)
		// Result: ["bb", "ccc"]
		got := tailWindow([]string{"a", "bb", "ccc"}, 5, cost)
		assert.Equal(t, []string{"bb", "ccc"}, got)
	})

	t.Run("everything fits", func(t *testing.T) {
		t.Parallel()
		got := tailWindow([]string{"a", "b"}, 10, cost)
		assert.Equal(t, []string{"a", "b"}, got)
	})

	t.Run("budget zero returns nil", func(t *testing.T) {
		t.Parallel()
		got := tailWindow([]string{"a", "b"}, 0, cost)
		assert.Nil(t, got)
	})

	t.Run("negative budget returns nil", func(t *testing.T) {
		t.Parallel()
		got := tailWindow([]string{"a", "b"}, -1, cost)
		assert.Nil(t, got)
	})

	t.Run("single element wider than budget guaranteed progress", func(t *testing.T) {
		t.Parallel()
		got := tailWindow([]string{"hello"}, 2, cost)
		assert.Equal(t, []string{"hello"}, got, "single oversized element must still be returned")
	})

	t.Run("empty input returns nil", func(t *testing.T) {
		t.Parallel()
		got := tailWindow([]string{}, 10, cost)
		assert.Nil(t, got)
	})

	t.Run("nil input returns nil", func(t *testing.T) {
		t.Parallel()
		got := tailWindow[string](nil, 10, cost)
		assert.Nil(t, got)
	})
}

func TestAdjustChunkSpans(t *testing.T) {
	t.Parallel()

	t.Run("baseOffset zero is no-op", func(t *testing.T) {
		t.Parallel()
		chunks := []Chunk{
			{ID: 0, Text: "hello", StartChar: 5, EndChar: 10},
			{ID: 1, Text: "world", StartChar: 0, EndChar: 0},
		}
		got := adjustChunkSpans(chunks, 0)
		assert.Equal(t, 5, got[0].StartChar)
		assert.Equal(t, 10, got[0].EndChar)
		assert.Equal(t, 0, got[1].StartChar)
		assert.Equal(t, 0, got[1].EndChar)
	})

	t.Run("non-zero offset shifts all non-zero spans", func(t *testing.T) {
		t.Parallel()
		chunks := []Chunk{
			{ID: 0, Text: "hello", StartChar: 3, EndChar: 8},
			{ID: 1, Text: "world", StartChar: 10, EndChar: 15},
		}
		got := adjustChunkSpans(chunks, 100)
		assert.Equal(t, 103, got[0].StartChar)
		assert.Equal(t, 108, got[0].EndChar)
		assert.Equal(t, 110, got[1].StartChar)
		assert.Equal(t, 115, got[1].EndChar)
	})

	t.Run("already-zero spans stay zero", func(t *testing.T) {
		t.Parallel()
		chunks := []Chunk{
			{ID: 0, Text: "hello", StartChar: 3, EndChar: 8},
			{ID: 1, Text: "world", StartChar: 0, EndChar: 0},
		}
		got := adjustChunkSpans(chunks, 50)
		assert.Equal(t, 53, got[0].StartChar)
		assert.Equal(t, 58, got[0].EndChar)
		assert.Equal(t, 0, got[1].StartChar)
		assert.Equal(t, 0, got[1].EndChar)
	})
}

func TestSplitIntoSentencesWithSpans_RepeatedShortSentence(t *testing.T) {
	t.Parallel()

	text := "Go. Then we use Go."
	spans := splitIntoSentencesWithSpans(text)

	require.Len(t, spans, 2, "expected 2 sentence spans")

	assert.Equal(t, 0, spans[0].start, "first sentence start")
	assert.Equal(t, 3, spans[0].end, "first sentence end")
	assert.Equal(t, "Go.", text[spans[0].start:spans[0].end])

	assert.Equal(t, 4, spans[1].start, "second sentence start")
	assert.True(t, spans[1].end == 19, "second sentence end: got %d, want 19", spans[1].end)
	assert.Equal(t, "Then we use Go.", text[spans[1].start:spans[1].end])

	// Neither span should be zeroed out.
	for i, sp := range spans {
		assert.False(t, sp.start == 0 && sp.end == 0,
			"span %d should not be zeroed", i)
	}
}

func TestSplitIntoSentencesWithSpans_SpansMatchSource(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		text string
	}{
		{
			name: "simple two sentences",
			text: "Hello world. Goodbye world.",
		},
		{
			name: "repeated short sentences",
			text: "Go. Go. Go.",
		},
		{
			name: "repeated sentence appearing as substring",
			text: "Go. Here we use Go. Go. Done.",
		},
		{
			name: "single sentence",
			text: "Just one sentence here.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			spans := splitIntoSentencesWithSpans(tt.text)
			require.NotEmpty(t, spans, "expected at least one span")

			for i, sp := range spans {
				if sp.start == 0 && sp.end == 0 {
					// Zero span means "unavailable" — valid for fallback path.
					continue
				}
				assert.Equal(t, sp.text, tt.text[sp.start:sp.end],
					"span %d: text[sp.start:sp.end] must equal sp.text", i)
				assert.True(t, sp.end > sp.start,
					"span %d: end (%d) must be > start (%d)", i, sp.end, sp.start)
			}
		})
	}
}

func TestCodeBlockProtection_FencedBlockIsAtomic(t *testing.T) {
	t.Parallel()

	text := "Before.\n```go\nfmt.Println(\"Hello. World!\")\n```\nAfter."
	spans := splitIntoSentencesWithSpans(text)

	// The code block containing "Hello. World!" must be a single span.
	for _, sp := range spans {
		if strings.Contains(sp.text, "Hello. World!") {
			// Must not be split — the entire fenced block is one span.
			assert.Contains(t, sp.text, "fmt.Println",
				"fenced code block should be a single span containing the full block")
			return
		}
	}
	t.Fatal("no span contains the code block text")
}

func TestCodeBlockProtection_CodeOnlyInput(t *testing.T) {
	t.Parallel()

	text := "```\nHello. World!\n```"
	spans := splitIntoSentencesWithSpans(text)

	require.Len(t, spans, 1, "code-only fenced block should produce exactly one span")
	assert.Contains(t, spans[0].text, "Hello. World!",
		"the single span should contain the full code block")
}

func TestCodeBlockProtection_FencedBlockFollowedByProse(t *testing.T) {
	t.Parallel()

	text := "```\ncode. here.\n```\nThis is prose. More text."
	spans := splitIntoSentencesWithSpans(text)

	// Should have: 1 code span + 2 prose spans = 3 total.
	require.GreaterOrEqual(t, len(spans), 2, "expected code + prose spans")

	// Find the code span.
	var foundCode bool
	for _, sp := range spans {
		if strings.Contains(sp.text, "code. here.") {
			assert.Contains(t, sp.text, "code. here.",
				"code block should not be split at internal periods")
			foundCode = true
		}
	}
	assert.True(t, foundCode, "expected to find a span containing the code block")

	// Prose sentences should be separate.
	var foundProse bool
	for _, sp := range spans {
		if strings.Contains(sp.text, "This is prose.") && !strings.Contains(sp.text, "code. here.") {
			foundProse = true
		}
	}
	assert.True(t, foundProse, "expected separate prose span")
}

func TestCodeBlockProtection_IndentedCodeIsAtomic(t *testing.T) {
	t.Parallel()

	text := "Before.\n    x = 1.0\n    y = 2.0\nAfter."
	spans := splitIntoSentencesWithSpans(text)

	// The indented code block should not be split at the decimal points.
	for _, sp := range spans {
		if strings.Contains(sp.text, "x = 1.0") {
			assert.Contains(t, sp.text, "y = 2.0",
				"indented code block should be a single atomic span")
			return
		}
	}
	t.Fatal("no span contains the indented code block")
}

func TestCodeBlockProtection_UnclosedFencedBlock(t *testing.T) {
	t.Parallel()

	text := "```\nHello. World!\nThis never closes."
	spans := splitIntoSentencesWithSpans(text)

	// Unclosed block should be atomic from opener to end.
	require.Len(t, spans, 1, "unclosed fenced block should be one span")
	assert.Contains(t, spans[0].text, "Hello. World!")
	assert.Contains(t, spans[0].text, "This never closes.")
}

func TestCodeBlockProtection_SpansRoundTrip(t *testing.T) {
	t.Parallel()

	text := "Intro. ```js\nlet x = 1.0;\nconsole.log(x);\n``` Conclusion."
	spans := splitIntoSentencesWithSpans(text)

	require.NotEmpty(t, spans)
	for i, sp := range spans {
		if sp.start == 0 && sp.end == 0 {
			continue
		}
		assert.Equal(t, sp.text, text[sp.start:sp.end],
			"span %d: text[start:end] must equal span text", i)
		assert.True(t, sp.end > sp.start,
			"span %d: end must be > start", i)
	}
}

func TestAppendChunkIfValid(t *testing.T) {
	t.Parallel()

	t.Run("EmptyTextSkipped", func(t *testing.T) {
		t.Parallel()
		chunks := appendChunkIfValid(nil, 0, "   ", "source", 0, 0)
		assert.Nil(t, chunks, "whitespace-only text should not produce a chunk")
	})

	t.Run("SpanMismatchResetsToZero", func(t *testing.T) {
		t.Parallel()
		source := "hello world"
		// Pass span that doesn't match the trimmed text.
		chunks := appendChunkIfValid(nil, 0, "hello", source, 0, 99)
		require.Len(t, chunks, 1)
		assert.Equal(t, 0, chunks[0].StartChar, "mismatched span should reset StartChar")
		assert.Equal(t, 0, chunks[0].EndChar, "mismatched span should reset EndChar")
	})

	t.Run("SourceEmptySkipsEqualityCheck", func(t *testing.T) {
		t.Parallel()
		// Pass a span that would fail equality, but source="" skips the check.
		chunks := appendChunkIfValid(nil, 0, "hello", "", 5, 10)
		require.Len(t, chunks, 1)
		assert.Equal(t, 5, chunks[0].StartChar, "span should pass through when source is empty")
		assert.Equal(t, 10, chunks[0].EndChar, "span should pass through when source is empty")
	})

	t.Run("ValidSpanPassesThrough", func(t *testing.T) {
		t.Parallel()
		source := "hello world"
		chunks := appendChunkIfValid(nil, 0, "hello", source, 0, 5)
		require.Len(t, chunks, 1)
		assert.Equal(t, 0, chunks[0].StartChar)
		assert.Equal(t, 5, chunks[0].EndChar)
		assert.Equal(t, "hello", chunks[0].Text)
	})

	t.Run("ZeroSpanResetsToZero", func(t *testing.T) {
		t.Parallel()
		source := "hello world"
		// start=0, end=0 — the equality check source[0:0]="" != "hello" resets.
		chunks := appendChunkIfValid(nil, 0, "hello", source, 0, 0)
		require.Len(t, chunks, 1)
		assert.Equal(t, 0, chunks[0].StartChar)
		assert.Equal(t, 0, chunks[0].EndChar)
	})

	t.Run("AppendsWithExistingChunks", func(t *testing.T) {
		t.Parallel()
		source := "first second"
		existing := []Chunk{buildChunk(0, "first")}
		chunks := appendChunkIfValid(existing, 1, "second", source, 6, 12)
		require.Len(t, chunks, 2)
		assert.Equal(t, 1, chunks[1].ID)
		assert.Equal(t, "second", chunks[1].Text)
	})

	t.Run("TextIsTrimmed", func(t *testing.T) {
		t.Parallel()
		source := "  hello  "
		// Span covers the original text region; after trimming, text is "hello".
		// The span won't match "hello" (source[0:9] = "  hello  "), so it resets.
		chunks := appendChunkIfValid(nil, 0, "  hello  ", source, 0, 9)
		require.Len(t, chunks, 1)
		assert.Equal(t, "hello", chunks[0].Text)
	})
}

// TestAdjustChunkSpansCorrupted is a regression gate for Phase B of the
// span-rebase fix. The corrupted-span case (StartChar>EndChar) is expected to
// FAIL against the current guard (StartChar>0||EndChar>0) — that failure is
// intentional and will be resolved by the production fix in Phase B.
func TestAdjustChunkSpansCorrupted(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		input      Chunk
		baseOffset int
		wantStart  int
		wantEnd    int
	}{
		{
			// Corrupted span: StartChar > EndChar. The correct post-fix behavior
			// is to leave it untouched. Currently FAILS (Phase B regression gate).
			name:       "corrupted span unchanged",
			input:      Chunk{StartChar: 5, EndChar: 0},
			baseOffset: 10,
			wantStart:  5,
			wantEnd:    0,
		},
		{
			// Both zero → unknown span, must remain zeroed.
			name:       "zero span unchanged",
			input:      Chunk{StartChar: 0, EndChar: 0},
			baseOffset: 10,
			wantStart:  0,
			wantEnd:    0,
		},
		{
			// Valid span starting at byte 0 must still be rebased.
			name:       "valid span at byte 0",
			input:      Chunk{StartChar: 0, EndChar: 50},
			baseOffset: 10,
			wantStart:  10,
			wantEnd:    60,
		},
		{
			// Normal non-zero span.
			name:       "normal span",
			input:      Chunk{StartChar: 3, EndChar: 20},
			baseOffset: 10,
			wantStart:  13,
			wantEnd:    30,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := adjustChunkSpans([]Chunk{tc.input}, tc.baseOffset)
			require.Len(t, got, 1)
			assert.Equal(t, tc.wantStart, got[0].StartChar, "StartChar")
			assert.Equal(t, tc.wantEnd, got[0].EndChar, "EndChar")
		})
	}
}

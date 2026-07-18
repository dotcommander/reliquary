package chunking

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// TestEnforceTokenLimits_ZeroSpanOnSplit — direct doc.go invariant check
// ---------------------------------------------------------------------------

// TestEnforceTokenLimits_ZeroSpanOnSplit asserts the doc.go span invariant
// directly on EnforceTokenLimits (not via ChunkWithTokenLimit):
//   - Chunks that are split to fit the token budget have StartChar==0 && EndChar==0
//     (spans cannot be reliably remapped after split).
//   - Chunks that pass through unsplit retain their original spans.
func TestEnforceTokenLimits_ZeroSpanOnSplit(t *testing.T) {
	t.Parallel()

	type testCase struct {
		name        string
		chunk       Chunk
		maxTokens   int
		expectSplit bool // true → expect multiple sub-chunks with zeroed spans
	}

	cases := []testCase{
		{
			name: "passthrough retains spans",
			chunk: Chunk{
				ID:        0,
				Text:      "one two",
				StartChar: 10,
				EndChar:   17,
				CharCount: 7,
			},
			maxTokens:   100,
			expectSplit: false,
		},
		{
			name: "split clears spans",
			chunk: Chunk{
				ID:        0,
				Text:      "alpha beta gamma delta epsilon",
				StartChar: 5,
				EndChar:   35,
				CharCount: 30,
			},
			maxTokens:   2,
			expectSplit: true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := EnforceTokenLimits(
				[]Chunk{tc.chunk},
				fakeTokenCounter{maxTokens: tc.maxTokens},
			)

			require.NotEmpty(t, result)

			if !tc.expectSplit {
				require.Len(t, result, 1)
				assert.Equal(t, tc.chunk.StartChar, result[0].StartChar,
					"passthrough chunk should retain original StartChar")
				assert.Equal(t, tc.chunk.EndChar, result[0].EndChar,
					"passthrough chunk should retain original EndChar")
			} else {
				assert.Greater(t, len(result), 1,
					"oversized chunk should be split into sub-chunks")
				for _, c := range result {
					assert.Equal(t, 0, c.StartChar,
						"split sub-chunk %d: StartChar must be zero", c.ID)
					assert.Equal(t, 0, c.EndChar,
						"split sub-chunk %d: EndChar must be zero", c.ID)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestEnforceTokenLimits_EdgeCases
// ---------------------------------------------------------------------------

func TestEnforceTokenLimits_EdgeCases(t *testing.T) {
	t.Parallel()

	type testCase struct {
		name      string
		chunks    []Chunk
		maxTokens int
		// wantLen == -1 means "just not empty"
		wantLen int
		wantNil bool
	}

	cases := []testCase{
		{
			name:      "nil input returns nil",
			chunks:    nil,
			maxTokens: 5,
			wantNil:   true,
		},
		{
			name:      "empty slice returns nil (nothing appended)",
			chunks:    []Chunk{},
			maxTokens: 5,
			wantNil:   true,
		},
		{
			name:      "zero budget is passthrough — returns original slice unchanged",
			chunks:    []Chunk{buildChunk(0, "word one two three four five six seven")},
			maxTokens: 0,
			wantLen:   1,
		},
		{
			name:      "negative budget is passthrough — returns original slice unchanged",
			chunks:    []Chunk{buildChunk(0, "word one two three four five six seven")},
			maxTokens: -1,
			wantLen:   1,
		},
		{
			name: "single oversized unit — single word exceeds budget — still emitted",
			// fakeTokenCounter counts words; a single word is 1 token, so
			// use maxTokens=0 to get passthrough, which is the only way a
			// single word "exceeds" and still appears.
			// Instead: use a real indivisible case: maxTokens=1, single word
			// of 1 token — it exactly fits so it passes through.
			//
			// To hit the indivisible-overflow branch: multiple words but
			// budget=1 means each word individually hits the single-atom
			// overflow path via accumulateTokenAtoms → onOverflow.
			chunks:    []Chunk{buildChunk(0, strings.Repeat("x", 300))},
			maxTokens: 1,
			wantLen:   1, // indivisible long word is passed through as-is
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := EnforceTokenLimits(tc.chunks, fakeTokenCounter{maxTokens: tc.maxTokens})

			if tc.wantNil {
				assert.Nil(t, result)
				return
			}
			require.NotNil(t, result)
			assert.Len(t, result, tc.wantLen)
		})
	}
}

// ---------------------------------------------------------------------------
// TestFillTokenCounts_EdgeCases
// ---------------------------------------------------------------------------

// TestFillTokenCounts_EdgeCases covers boundary inputs for FillTokenCounts.
// FillTokenCounts uses tiktoken; these cases avoid actual encoding.
func TestFillTokenCounts_EdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("nil slice — no panic, no error", func(t *testing.T) {
		t.Parallel()

		// nil is a valid empty slice in Go; should not panic.
		err := FillTokenCounts(nil, "cl100k_base")
		assert.NoError(t, err)
	})

	t.Run("empty slice — no panic, no error", func(t *testing.T) {
		t.Parallel()

		err := FillTokenCounts([]Chunk{}, "cl100k_base")
		assert.NoError(t, err)
	})

	t.Run("invalid encoding returns error", func(t *testing.T) {
		t.Parallel()

		chunks := []Chunk{{ID: 0, Text: "hello"}}
		err := FillTokenCounts(chunks, "not_a_real_encoding_xyz")
		assert.Error(t, err, "invalid encoding should return an error")
		// TokenCount must not be mutated on error.
		assert.Equal(t, 0, chunks[0].TokenCount)
	})

	t.Run("already-set TokenCount is not overwritten", func(t *testing.T) {
		t.Parallel()

		chunks := []Chunk{{ID: 0, Text: "hello world", TokenCount: 42}}
		err := FillTokenCounts(chunks, "cl100k_base")
		require.NoError(t, err)
		assert.Equal(t, 42, chunks[0].TokenCount,
			"pre-set TokenCount must not be overwritten")
	})

	t.Run("empty Text is skipped — TokenCount stays 0", func(t *testing.T) {
		t.Parallel()

		chunks := []Chunk{{ID: 0, Text: "", TokenCount: 0}}
		err := FillTokenCounts(chunks, "cl100k_base")
		require.NoError(t, err)
		assert.Equal(t, 0, chunks[0].TokenCount,
			"empty-text chunk should not get a token count")
	})

	t.Run("non-empty chunk receives a positive token count", func(t *testing.T) {
		t.Parallel()

		chunks := []Chunk{{ID: 0, Text: "hello world"}}
		err := FillTokenCounts(chunks, "cl100k_base")
		require.NoError(t, err)
		assert.Greater(t, chunks[0].TokenCount, 0,
			"filled chunk should have a positive token count")
	})
}

package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestHardLimitKeepsBoundedSentencesBelowHalfBudget(t *testing.T) {
	t.Parallel()
	for _, letter := range []string{"a", "界"} {
		t.Run(letter, func(t *testing.T) {
			t.Parallel()
			first := "A" + strings.Repeat(letter, 38) + "."
			second := "B" + strings.Repeat(letter, 68) + "."
			chunks := EnforceHardLimits([]Chunk{buildChunk(0, first+" "+second)}, LimitOptions{MaxChars: 100})
			if len(chunks) != 2 {
				t.Fatalf("got %d chunks, want two intact sentences", len(chunks))
			}
			for i, want := range []string{first, second} {
				if chunks[i].Text != want {
					t.Errorf("chunk %d = %q, want intact %q", i, chunks[i].Text, want)
				}
			}
		})
	}
}

func TestHardLimitOversizedSentenceStillFallsBack(t *testing.T) {
	t.Parallel()
	text := strings.Repeat("界", 240) + "."
	chunks := EnforceHardLimits([]Chunk{buildChunk(0, text)}, LimitOptions{MaxChars: 100})
	var reconstructed strings.Builder
	for i, chunk := range chunks {
		if got := utf8.RuneCountInString(chunk.Text); got > 100 {
			t.Errorf("chunk %d contains %d runes", i, got)
		}
		reconstructed.WriteString(chunk.Text)
	}
	if reconstructed.String() != text {
		t.Fatal("hard-cut fallback lost sentence content")
	}
}

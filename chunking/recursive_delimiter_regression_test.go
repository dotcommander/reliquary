package chunking

import (
	"strings"
	"testing"
)

func TestRecursiveDelimiterRestartRegression(t *testing.T) {
	t.Parallel()
	source := "猫猫||犬犬||鳥鳥"
	chunker := NewRecursiveChunker(SeparatorProfile{Separators: []string{"||"}})
	chunks := chunker.Chunk(source, 3, 0)
	if len(chunks) != 3 {
		t.Fatalf("delimiter caused extra split: %#v", chunks)
	}
	for _, c := range chunks {
		if strings.Contains(c.Text, "|") {
			t.Fatalf("delimiter leaked: %#v", c)
		}
	}
	assertRegressionChunkBounds(t, source, chunks, 3)
	overlapped := chunker.Chunk(source, 4, 1)
	if len(overlapped) == 0 {
		t.Fatal("missing positive-overlap content")
	}
	assertRegressionChunkBounds(t, source, overlapped, 4)
}

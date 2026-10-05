package chunking

import "testing"

func TestParagraphExactBudgetRegression(t *testing.T) {
	t.Parallel()
	for _, source := range []string{"abc def", "猫猫 犬犬"} {
		size := len([]rune(source))
		chunks := newParagraphAwareChunker().Chunk(source, size, 0)
		if len(chunks) != 1 || chunks[0].Text != source {
			t.Fatalf("first paragraph paid nonexistent separator: %#v", chunks)
		}
		assertRegressionChunkBounds(t, source, chunks, size)
	}
	source := "猫猫\n\n犬犬"
	chunks := newParagraphAwareChunker().Chunk(source, 6, 0)
	if len(chunks) != 1 || chunks[0].Text != source {
		t.Fatalf("joined paragraph = %#v", chunks)
	}
	assertRegressionChunkBounds(t, source, chunks, 6)
}

func TestParagraphUnknownSpanRegression(t *testing.T) {
	t.Parallel()
	cases := []struct {
		spans              []textSpan
		wantStart, wantEnd int
	}{
		{[]textSpan{{start: 0, end: 3}, {start: 5, end: 8}}, 0, 8},
		{[]textSpan{{start: 0, end: 0}, {start: 5, end: 8}}, 0, 0},
		{[]textSpan{{start: 0, end: 3}, {start: 0, end: 0}, {start: 8, end: 11}}, 0, 0},
		{[]textSpan{{start: 2, end: 5}, {start: 0, end: 0}}, 0, 0},
	}
	for _, tc := range cases {
		start, end := mergeParaSpans(tc.spans, 0, len(tc.spans)-1)
		if start != tc.wantStart || end != tc.wantEnd {
			t.Fatalf("span = %d,%d, want %d,%d", start, end, tc.wantStart, tc.wantEnd)
		}
	}
	source := "abc \n\n def"
	chunks := newParagraphAwareChunker().Chunk(source, 20, 0)
	assertRegressionChunkBounds(t, source, chunks, 20)
}

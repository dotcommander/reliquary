package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func assertRegressionChunkBounds(t *testing.T, source string, chunks []Chunk, size int) {
	t.Helper()
	for _, c := range chunks {
		if !utf8.ValidString(c.Text) || utf8.RuneCountInString(c.Text) > size {
			t.Fatalf("invalid or oversized chunk: %#v", c)
		}
		if c.StartChar != 0 || c.EndChar != 0 {
			if c.StartChar < 0 || c.EndChar <= c.StartChar || c.EndChar > len(source) || source[c.StartChar:c.EndChar] != c.Text {
				t.Fatalf("inexact byte span: %#v", c)
			}
		}
	}
}

func TestUnicodeBoundaryBudgetsRegression(t *testing.T) {
	t.Parallel()
	for _, chunker := range []Chunker{newSentenceBoundaryChunker(), newWordBoundaryChunker(), newSmartBoundaryChunker()} {
		t.Run(string(chunker.Strategy()), func(t *testing.T) {
			t.Parallel()
			for _, source := range []string{"猫猫。 犬犬。 鳥鳥。", "猫猫 犬犬 鳥鳥", "One. Two. Three."} {
				for _, overlap := range []int{0, 3, 5, 8} {
					chunks := chunker.Chunk(source, 8, overlap)
					if len(chunks) == 0 {
						t.Fatal("missing content")
					}
					assertRegressionChunkBounds(t, source, chunks, 8)
					if overlap == 0 {
						for _, word := range strings.Fields(source) {
							if !strings.Contains(strings.Join(chunkTextsRegression(chunks), " "), word) {
								t.Fatalf("lost word %q", word)
							}
						}
					}
				}
			}
		})
	}
}

func chunkTextsRegression(chunks []Chunk) []string {
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	return texts
}

func TestUnicodeOverlapCostsRegression(t *testing.T) {
	t.Parallel()
	var b strings.Builder
	writeWordOverlap(&b, 1, 3, []string{"猫猫"})
	if b.String() != "猫猫 " {
		t.Fatalf("word overlap = %q", b.String())
	}
	b.Reset()
	writeSmartOverlap(&b, 1, 3, []string{"猫猫"})
	if b.String() != "猫猫 " {
		t.Fatalf("smart overlap = %q", b.String())
	}
	b.Reset()
	writeSentenceOverlap(&b, 1, 3, []string{"猫猫"})
	if b.String() != "猫猫 " {
		t.Fatalf("sentence overlap = %q", b.String())
	}
	b.Reset()
	writeSentenceOverlap(&b, 1, 1, []string{"猫猫"})
	if b.Len() != 0 {
		t.Fatal("overlap exceeded rune budget")
	}
	for _, chunker := range []Chunker{newSentenceBoundaryChunker(), newWordBoundaryChunker(), newSmartBoundaryChunker()} {
		chunks := chunker.Chunk("猫猫", 2, 0)
		if len(chunks) != 1 || chunks[0].Text != "猫猫" {
			t.Fatalf("rune-sized unit split: %#v", chunks)
		}
	}
}

func TestBoundaryASCIIParityRegression(t *testing.T) {
	t.Parallel()
	const source = "One. Two. Three."
	for _, chunker := range []Chunker{newSentenceBoundaryChunker(), newWordBoundaryChunker(), newSmartBoundaryChunker()} {
		chunks := chunker.Chunk(source, len(source), 0)
		if len(chunks) != 1 || chunks[0].Text != source || chunks[0].StartChar != 0 || chunks[0].EndChar != len(source) {
			t.Fatalf("%s changed fitting ASCII input: %#v", chunker.Strategy(), chunks)
		}
	}
}

package chunking

import (
	"strings"
	"unicode/utf8"
)

// sentenceBoundaryChunker splits text at sentence boundaries.
type sentenceBoundaryChunker struct{}

func newSentenceBoundaryChunker() *sentenceBoundaryChunker {
	return &sentenceBoundaryChunker{}
}

func (s *sentenceBoundaryChunker) Strategy() Strategy {
	return SentenceBoundary
}

func (s *sentenceBoundaryChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}

	spans := splitIntoSentencesWithSpans(text)
	sentences := make([]string, len(spans))
	for i, sp := range spans {
		sentences[i] = sp.text
	}

	chunks := buildSentenceChunksWithSpans(text, spans, sentences, size, overlap, sentenceChunkOps{
		writeOverlap: writeSentenceOverlap,
		fill:         fillSentenceChunk,
		makeOverlap:  func(added []string, _ int) []string { return added },
	})
	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size, Overlap: overlap, OriginalText: text})
}

type sentenceChunkOps struct {
	writeOverlap func(*strings.Builder, int, int, []string)
	fill         func(*strings.Builder, []string, int, int, int) (int, []string)
	makeOverlap  func([]string, int) []string
}

// buildSentenceChunksWithSpans builds sentence chunks populating
// StartChar/EndChar byte offsets from the sentence spans.
func buildSentenceChunksWithSpans(source string, spans []textSpan, sentences []string, size, overlap int, ops sentenceChunkOps) []Chunk {
	var chunks []Chunk
	var currentChunk strings.Builder
	var overlapSentences []string
	chunkID := 0

	for i := 0; i < len(sentences); {
		currentChunk.Reset()

		ops.writeOverlap(&currentChunk, chunkID, overlap, overlapSentences)

		startLen := currentChunk.Len()

		// Escape hatch: if a single sentence exceeds size and no overlap
		// was pre-filled, emit it as a standalone chunk to avoid the
		// fill-then-force-add cycle producing chunks with overlap when
		// none is appropriate.
		if i < len(sentences) && len(sentences[i]) > size && currentChunk.Len() == 0 {
			var startChar, endChar int
			if i < len(spans) && (spans[i].start > 0 || spans[i].end > 0) {
				startChar = spans[i].start
				endChar = spans[i].end
			}
			overlapSentences = []string{sentences[i]}
			prevLen := len(chunks)
			chunks = appendChunkIfValid(chunks, chunkID, sentences[i], "", startChar, endChar)
			if len(chunks) > prevLen {
				chunkID++
			}
			i++
			continue
		}

		var added []string
		var addedStartIdx int // first sentence index added to this chunk
		i, added = ops.fill(&currentChunk, sentences, i, size, startLen)
		if len(added) > 0 {
			addedStartIdx = i - len(added)
		}
		overlapSentences = ops.makeOverlap(added, overlap)

		// If nothing was added, force-add one sentence.
		if currentChunk.Len() == startLen && i < len(sentences) {
			addedStartIdx = i
			currentChunk.WriteString(sentences[i])
			overlapSentences = []string{sentences[i]}
			i++
		}

		chunkText := currentChunk.String()
		// Determine byte span from the sentence spans.
		startChar, endChar := 0, 0
		if len(added) > 0 && addedStartIdx < len(spans) {
			endIdx := addedStartIdx + len(added) - 1
			if endIdx >= len(spans) {
				endIdx = len(spans) - 1
			}
			if spans[addedStartIdx].start > 0 || spans[addedStartIdx].end > 0 {
				startChar = spans[addedStartIdx].start
			}
			if spans[endIdx].start > 0 || spans[endIdx].end > 0 {
				endChar = spans[endIdx].end
			}
		}

		prevLen := len(chunks)
		chunks = appendChunkIfValid(chunks, chunkID, chunkText, source, startChar, endChar)
		if len(chunks) > prevLen {
			chunkID++
		}
	}

	return chunks
}

func writeSentenceOverlap(builder *strings.Builder, chunkID, overlap int, sentences []string) {
	if chunkID == 0 || overlap <= 0 || len(sentences) == 0 {
		return
	}
	window := tailWindow(sentences, overlap, func(s string) int { return len(s) + 1 })
	for _, sent := range window {
		builder.WriteString(sent)
		builder.WriteString(" ")
	}
}

func fillSentenceChunk(builder *strings.Builder, sentences []string, i, size, startLen int) (int, []string) {
	added := make([]string, 0)
	runeCount := utf8.RuneCountInString(builder.String())
	for i < len(sentences) && runeCount+utf8.RuneCountInString(sentences[i]) <= size {
		builder.WriteString(sentences[i])
		runeCount += utf8.RuneCountInString(sentences[i])
		if !strings.HasSuffix(sentences[i], " ") {
			builder.WriteString(" ")
			runeCount++
		}
		added = append(added, sentences[i])
		i++
	}
	return i, added
}

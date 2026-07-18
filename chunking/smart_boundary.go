package chunking

import (
	"strings"
	"unicode/utf8"
)

// smartBoundaryChunker uses fast regex-based sentence detection to split text
// at natural boundaries with overlap support.
type smartBoundaryChunker struct{}

func newSmartBoundaryChunker() *smartBoundaryChunker {
	return &smartBoundaryChunker{}
}

func (s *smartBoundaryChunker) Strategy() Strategy {
	return SmartBoundary
}

func (s *smartBoundaryChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}

	spans := splitIntoSentencesWithSpans(text)
	sentences := make([]string, len(spans))
	for i, sp := range spans {
		sentences[i] = sp.text
	}

	chunks := buildSentenceChunksWithSpans(text, spans, sentences, size, overlap, sentenceChunkOps{
		writeOverlap: writeSmartOverlap,
		fill:         fillSmartSentenceChunk,
		makeOverlap:  tailSentenceOverlap,
	})
	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size, Overlap: overlap, OriginalText: text})
}

func writeSmartOverlap(builder *strings.Builder, chunkID, overlap int, sentences []string) {
	if chunkID == 0 || overlap <= 0 || len(sentences) == 0 {
		return
	}
	overlapText := strings.Join(sentences, " ")
	if len(overlapText) <= overlap {
		builder.WriteString(overlapText)
		builder.WriteString(" ")
	}
}

func fillSmartSentenceChunk(builder *strings.Builder, sentences []string, i, size, startLen int) (int, []string) {
	added := make([]string, 0)
	runeCount := utf8.RuneCountInString(builder.String())
	for i < len(sentences) && runeCount+utf8.RuneCountInString(sentences[i]) <= size {
		if builder.Len() > startLen {
			builder.WriteString(" ")
			runeCount++
		}
		builder.WriteString(sentences[i])
		runeCount += utf8.RuneCountInString(sentences[i])
		added = append(added, sentences[i])
		i++
	}
	return i, added
}

// tailSentenceOverlap returns the trailing sentences whose total byte length
// fits within the overlap budget. Uses tailWindow for selection.
func tailSentenceOverlap(sentences []string, overlap int) []string {
	return tailWindow(sentences, overlap, func(s string) int { return len(s) })
}

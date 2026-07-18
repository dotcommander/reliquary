package chunking

import (
	"strings"
	"unicode/utf8"
)

// paragraphAwareChunker splits text at paragraph boundaries,
// falling back to word-boundary splitting for oversized paragraphs.
type paragraphAwareChunker struct{}

func newParagraphAwareChunker() *paragraphAwareChunker {
	return &paragraphAwareChunker{}
}

func (p *paragraphAwareChunker) Strategy() Strategy {
	return ParagraphAware
}

func (p *paragraphAwareChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}

	paragraphs := splitIntoParagraphs(text)
	// Build byte spans for each paragraph by searching in original text.
	paraSpans := locateTextSpans(text, paragraphs)

	var chunks []Chunk
	var currentChunk strings.Builder
	chunkID := 0

	for i := 0; i < len(paragraphs); {
		currentChunk.Reset()
		firstPara := i

		// Add paragraphs until the rune-size limit is reached.
		runeCount := utf8.RuneCountInString(currentChunk.String())
		for i < len(paragraphs) && runeCount+utf8.RuneCountInString(paragraphs[i])+2 <= size {
			if currentChunk.Len() > 0 {
				currentChunk.WriteString("\n\n")
				runeCount += 2
			}
			currentChunk.WriteString(paragraphs[i])
			runeCount += utf8.RuneCountInString(paragraphs[i])
			i++
		}

		// No paragraph was added — split oversized by words.
		// Escape hatch: oversized paragraph that doesn't fit in the builder.
		// Delegate to word-boundary chunker for sub-splitting.
		if currentChunk.Len() == 0 && i < len(paragraphs) {
			wordChunker := newWordBoundaryChunker()
			subChunks := wordChunker.Chunk(paragraphs[i], size, overlap)
			// Rebase sub-chunk spans from paragraph-relative to original-text-relative.
			if i < len(paraSpans) {
				subChunks = adjustChunkSpans(subChunks, paraSpans[i].start)
			}
			for _, sc := range subChunks {
				sc.ID = chunkID
				chunks = append(chunks, sc)
				chunkID++
			}
			i++
			continue
		}

		chunkText := strings.TrimSpace(currentChunk.String())
		if chunkText != "" {
			lastPara := i - 1
			startChar, endChar := mergeParaSpans(paraSpans, firstPara, lastPara)
			chunks = append(chunks, buildChunkWithSpan(chunkID, chunkText, startChar, endChar))
			chunkID++
		}
	}

	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size, Overlap: overlap, OriginalText: text})
}

// locateTextSpans finds byte offsets for each text fragment in the source.
func locateTextSpans(source string, fragments []string) []textSpan {
	spans := make([]textSpan, len(fragments))
	cursor := 0
	for i, frag := range fragments {
		idx := strings.Index(source[cursor:], frag)
		if idx >= 0 {
			spans[i] = textSpan{text: frag, start: cursor + idx, end: cursor + idx + len(frag)}
			cursor = spans[i].end
		} else {
			spans[i] = textSpan{text: frag, start: 0, end: 0}
		}
	}
	return spans
}

// mergeParaSpans returns the byte range covering paragraphs [first, last].
func mergeParaSpans(spans []textSpan, first, last int) (int, int) {
	if first >= len(spans) || last >= len(spans) {
		return 0, 0
	}
	start := spans[first].start
	end := spans[last].end
	// If any span is unknown, the merged span is unknown.
	if start == 0 && end == 0 {
		return 0, 0
	}
	return start, end
}

// splitIntoParagraphs splits text on double newlines.
func splitIntoParagraphs(text string) []string {
	parts := strings.Split(text, "\n\n")
	var paragraphs []string

	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			paragraphs = append(paragraphs, trimmed)
		}
	}

	// If no paragraphs were found, treat the whole text as one paragraph.
	if len(paragraphs) == 0 && strings.TrimSpace(text) != "" {
		paragraphs = []string{strings.TrimSpace(text)}
	}

	return paragraphs
}

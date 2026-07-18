package chunking

import (
	"strings"
	"unicode/utf8"
)

// wordBoundaryChunker splits text at word boundaries, preserving whitespace.
type wordBoundaryChunker struct{}

func newWordBoundaryChunker() *wordBoundaryChunker {
	return &wordBoundaryChunker{}
}

func (w *wordBoundaryChunker) Strategy() Strategy {
	return WordBoundary
}

func (w *wordBoundaryChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}

	words := splitIntoWords(text)
	var chunks []Chunk
	var currentChunk strings.Builder
	var overlapBuffer []string
	chunkID := 0

	// Build byte-offset map: cumulative byte length of concatenated words
	// equals the original text, so byte offset of word[i] = sum of len(word[0:i]).
	wordByteStarts := make([]int, len(words)+1)
	for i, w := range words {
		wordByteStarts[i+1] = wordByteStarts[i] + len(w)
	}

	for i := 0; i < len(words); {
		currentChunk.Reset()

		writeWordOverlap(&currentChunk, chunkID, overlap, overlapBuffer)

		// Clear overlap buffer for the new chunk.
		overlapBuffer = nil
		startPos := currentChunk.Len()

		// Escape hatch: single word wider than size and no overlap pre-filled.
		if i < len(words) && len(words[i]) > size && startPos == 0 {
			word := words[i]
			rawStart := wordByteStarts[i]
			rawEnd := wordByteStarts[i+1]
			startChar, endChar := trimSpanToText(text, word, rawStart, rawEnd)
			overlapBuffer = tailWordOverlapBuffer(word, overlap)
			chunks = append(chunks, buildChunkWithSpan(chunkID, word, startChar, endChar))
			chunkID++
			i++
			continue
		}

		firstWordIdx := i
		i = fillWordChunk(&currentChunk, words, i, size, startPos)
		lastWordIdx := i - 1

		chunkText := strings.TrimSpace(currentChunk.String())
		if chunkText != "" {
			// Byte span covers from the first word's start to the last word's end.
			// Adjust to match the trimmed text.
			rawStart := wordByteStarts[firstWordIdx]
			rawEnd := wordByteStarts[lastWordIdx+1]
			startChar, endChar := trimSpanToText(text, chunkText, rawStart, rawEnd)
			chunks = append(chunks, buildChunkWithSpan(chunkID, chunkText, startChar, endChar))
			overlapBuffer = tailWordOverlapBuffer(chunkText, overlap)
			chunkID++
		}
	}

	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size, Overlap: overlap, OriginalText: text})
}

func writeWordOverlap(builder *strings.Builder, chunkID, overlap int, buffer []string) {
	if chunkID == 0 || overlap <= 0 || len(buffer) == 0 {
		return
	}
	overlapText := strings.Join(buffer, " ")
	if len(overlapText) <= overlap {
		builder.WriteString(overlapText)
		builder.WriteString(" ")
	}
}

func fillWordChunk(builder *strings.Builder, words []string, i, size, startPos int) int {
	runeCount := utf8.RuneCountInString(builder.String())
	for i < len(words) && runeCount+utf8.RuneCountInString(words[i]) <= size {
		builder.WriteString(words[i])
		runeCount += utf8.RuneCountInString(words[i])
		i++
	}
	if builder.Len() == startPos && i < len(words) {
		builder.WriteString(words[i])
		i++
	}
	return i
}

// tailWordOverlapBuffer returns the trailing words from chunkText whose total
// byte length (including spaces) fits within overlap budget. Uses tailWindow
// to select the trailing window of words.
func tailWordOverlapBuffer(chunkText string, overlap int) []string {
	if overlap <= 0 {
		return nil
	}
	words := strings.Fields(chunkText)
	selected := tailWindow(words, overlap, func(s string) int { return len(s) + 1 }) // word + space
	if len(selected) == 0 {
		return nil
	}
	return selected
}

package chunking

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// defaultMaxChunkChars is the default maximum rune count per output chunk.
// Also used as the oversized-sentence fallback threshold in
// splitIntoSentencesWithRuneSpans: a single "sentence" longer than this is
// likely unpunctuated prose (lists, code, logs) and falls back to newline splitting.
const defaultMaxChunkChars = 1600

// textSpan pairs a text fragment with its byte offsets in the original input.
type textSpan struct {
	text  string
	start int // byte offset in original text
	end   int // byte offset in original text (exclusive)
}

// countWords returns the number of whitespace-delimited words in text.
func countWords(text string) int {
	return len(strings.Fields(text))
}

// splitIntoWords splits text into words and whitespace tokens,
// preserving whitespace for accurate boundary reconstruction.
func splitIntoWords(text string) []string {
	var words []string
	var currentWord strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) {
			if currentWord.Len() > 0 {
				words = append(words, currentWord.String())
				currentWord.Reset()
			}
			words = append(words, string(r))
		} else {
			currentWord.WriteRune(r)
		}
	}

	if currentWord.Len() > 0 {
		words = append(words, currentWord.String())
	}

	return words
}

// SplitSentences is the exported entry point for token-aware chunk validation in the actions layer.
func SplitSentences(text string) []string {
	return splitIntoSentences(text)
}

// splitIntoSentences splits text into sentences using punctuation-based detection.
// Handles common abbreviations (Mr., Dr., etc.) and decimal numbers to avoid
// false splits. ~1000x faster than prose NLP (~0.1ms vs ~130ms).
func splitIntoSentences(text string) []string {
	withSpans := splitIntoSentencesWithRuneSpans(text)
	sentences := make([]string, len(withSpans))
	for i, s := range withSpans {
		sentences[i] = s.text
	}
	return sentences
}

// splitIntoSentencesWithSpans splits text into sentences with byte offsets
// into the original text. Each span's [start, end) covers the trimmed
// sentence text as it appears in the original. Spans are (0, 0) when the
// sentence came from fallbackSplit and has no reliable rune span.
//
// Byte offsets are derived from rune-index positions captured during the
// punctuation scan, then converted via runeByteOffsets and trimSpanToText.
// This avoids the former strings.Index re-search which could misattribute
// spans when a sentence text appeared as a substring earlier in the document.
func splitIntoSentencesWithSpans(text string) []textSpan {
	withSpans := splitIntoSentencesWithRuneSpans(text)
	if len(withSpans) == 0 {
		return nil
	}

	runes := []rune(text)
	offsets := runeByteOffsets(text, len(runes))

	spans := make([]textSpan, len(withSpans))
	for i, s := range withSpans {
		if s.runeStart == 0 && s.runeEnd == 0 {
			// Fallback path: no rune span available.
			spans[i] = textSpan{text: s.text}
			continue
		}
		rawStart := offsets[s.runeStart]
		rawEnd := offsets[s.runeEnd]
		start, end := trimSpanToText(text, s.text, rawStart, rawEnd)
		spans[i] = textSpan{text: s.text, start: start, end: end}
	}
	return spans
}

func isSentenceTerminator(r rune) bool {
	return r == '.' || r == '!' || r == '?'
}

func isDecimalPoint(runes []rune, i int) bool {
	return runes[i] == '.' &&
		i > 0 &&
		i < len(runes)-1 &&
		unicode.IsDigit(runes[i-1]) &&
		unicode.IsDigit(runes[i+1])
}

func isKnownAbbreviation(runes []rune, i int, abbrevs map[string]bool) bool {
	if runes[i] != '.' || i == 0 {
		return false
	}
	wordStart := i - 1
	for wordStart > 0 && unicode.IsLetter(runes[wordStart-1]) {
		wordStart--
	}
	word := strings.ToLower(string(runes[wordStart:i]))
	return abbrevs[word]
}

// fallbackSplit splits text on double-newlines, then single newlines if needed.
// Used when punctuation-based sentence splitting produces a single oversized segment.
func fallbackSplit(text string) []string {
	// Try paragraph breaks first (double newlines).
	if paragraphs := splitIntoParagraphs(text); len(paragraphs) > 1 {
		return paragraphs
	}

	// Fall back to single newlines.
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			lines = append(lines, line)
		}
	}
	if len(lines) > 0 {
		return lines
	}

	return []string{text}
}

// runeByteOffsets builds a slice mapping rune index to byte offset in text.
// The returned slice has length n+1 where n is the number of runes in text;
// element i holds the byte offset of rune i, and element n holds len(text).
func runeByteOffsets(text string, n int) []int {
	offsets := make([]int, n+1)
	bytePos := 0
	for i := 0; i < n; i++ {
		offsets[i] = bytePos
		_, w := utf8.DecodeRuneInString(text[bytePos:])
		bytePos += w
	}
	offsets[n] = bytePos
	return offsets
}

// trimSpanToText adjusts a byte span [rawStart, rawEnd) in source so that
// source[adjustedStart:adjustedEnd] == target. It skips leading/trailing
// whitespace to match strings.TrimSpace semantics.
func trimSpanToText(source, target string, rawStart, rawEnd int) (int, int) {
	for rawStart < rawEnd && rawStart < len(source) && (source[rawStart] == ' ' || source[rawStart] == '\n' || source[rawStart] == '\t' || source[rawStart] == '\r') {
		rawStart++
	}
	for rawEnd > rawStart && rawEnd > 0 && (source[rawEnd-1] == ' ' || source[rawEnd-1] == '\n' || source[rawEnd-1] == '\t' || source[rawEnd-1] == '\r') {
		rawEnd--
	}
	return rawStart, rawEnd
}

// tailWindow returns the trailing slice of units whose total cost fits within
// budget. If the entire slice fits, it is returned as-is. If a single unit
// exceeds budget, that unit is still returned (guaranteed progress).
// T is the unit type; cost extracts the size contribution of one unit.
func tailWindow[T any](units []T, budget int, cost func(T) int) []T {
	if budget <= 0 || len(units) == 0 {
		return nil
	}
	total := 0
	start := len(units)
	for i := len(units) - 1; i >= 0; i-- {
		c := cost(units[i])
		if total+c > budget && start < len(units) {
			break
		}
		total += c
		start = i
	}
	return units[start:]
}

// appendChunkIfValid trims text, skips empty chunks, resets the byte span
// when source[start:end] != trimmed (overlap/trimming produced a mismatched
// span), then appends a new Chunk. Pass source="" to skip the span-equality
// check.
func appendChunkIfValid(chunks []Chunk, id int, text, source string, start, end int) []Chunk {
	text = strings.TrimSpace(text)
	if text == "" {
		return chunks
	}
	if source != "" && (end > len(source) || start >= end || source[start:end] != text) {
		start, end = 0, 0
	}
	return append(chunks, buildChunkWithSpan(id, text, start, end))
}

// adjustChunkSpans adds baseOffset to StartChar and EndChar on each chunk
// whose span is non-zero. Used to rebase sub-chunk spans from section-relative
// to original-text-relative coordinates. baseOffset == 0 is a no-op.
func adjustChunkSpans(chunks []Chunk, baseOffset int) []Chunk {
	if baseOffset == 0 {
		return chunks
	}
	for i := range chunks {
		if chunks[i].EndChar > chunks[i].StartChar {
			chunks[i].StartChar += baseOffset
			chunks[i].EndChar += baseOffset
		}
	}
	return chunks
}

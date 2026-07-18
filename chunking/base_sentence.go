package chunking

import (
	"strings"
	"unicode"
)

// sentenceWithRuneSpan pairs a sentence with its rune-index interval in the
// original text. runeStart/runeEnd are (0, 0) when the sentence came from
// fallbackSplit and has no reliable rune span.
type sentenceWithRuneSpan struct {
	text      string
	runeStart int // inclusive rune index in original text
	runeEnd   int // exclusive rune index in original text
}

// splitIntoSentencesWithRuneSpans performs the same punctuation-based sentence
// scan as splitIntoSentences but also captures the rune-index interval for each
// emitted sentence. Fallback sentences (from fallbackSplit) get (0, 0) spans.
// Fenced and indented code blocks are treated as atomic units — punctuation
// inside them does not trigger sentence breaks.
func splitIntoSentencesWithRuneSpans(text string) []sentenceWithRuneSpan {
	if text == "" {
		return nil
	}

	// Common abbreviations that shouldn't trigger sentence breaks.
	abbrevs := map[string]bool{
		"mr": true, "mrs": true, "ms": true, "dr": true, "prof": true,
		"sr": true, "jr": true, "st": true, "vs": true, "etc": true,
		"inc": true, "ltd": true, "corp": true, "dept": true, "gen": true,
		"gov": true, "sgt": true, "col": true, "capt": true, "rev": true,
		"fig": true, "vol": true, "no": true, "approx": true, "ca": true,
		"e.g": true, "i.e": true, "c": true,
	}

	runes := []rune(text)
	protected := findCodeBlockRanges(runes)
	var result []sentenceWithRuneSpan
	start := 0

	for i := 0; i < len(runes); {
		// If current position is inside a protected code block, emit it
		// as one atomic sentence and skip past the block.
		if blk := isInsideCodeBlock(protected, i); blk != nil {
			// Emit any prose before this code block.
			if i > start {
				prose := string(runes[start:i])
				result = append(result, extractSentenceSpans(prose, start, abbrevs)...)
			}
			// Emit the code block as a single unit.
			sent := strings.TrimSpace(string(runes[blk.runeStart:blk.runeEnd]))
			if sent != "" {
				result = append(result, sentenceWithRuneSpan{
					text:      sent,
					runeStart: blk.runeStart,
					runeEnd:   blk.runeEnd,
				})
			}
			start = blk.runeEnd
			i = blk.runeEnd
			continue
		}

		r := runes[i]
		if !isSentenceTerminator(r) {
			i++
			continue
		}

		if isDecimalPoint(runes, i) || isKnownAbbreviation(runes, i, abbrevs) {
			i++
			continue
		}

		// Must be followed by whitespace or end-of-text to be a sentence break.
		if i < len(runes)-1 && !unicode.IsSpace(runes[i+1]) {
			i++
			continue
		}

		sent := strings.TrimSpace(string(runes[start : i+1]))
		if sent != "" {
			result = append(result, sentenceWithRuneSpan{
				text:      sent,
				runeStart: start,
				runeEnd:   i + 1,
			})
		}
		start = i + 1
		i++
	}

	// Remaining text after last sentence-ending punctuation.
	if start < len(runes) {
		tail := strings.TrimSpace(string(runes[start:]))
		if tail != "" {
			result = append(result, sentenceWithRuneSpan{
				text:      tail,
				runeStart: start,
				runeEnd:   len(runes),
			})
		}
	}

	if len(result) == 0 {
		// Fallback: no rune spans available.
		fallback := fallbackSplit(text)
		out := make([]sentenceWithRuneSpan, len(fallback))
		for i, s := range fallback {
			out[i] = sentenceWithRuneSpan{text: s}
		}
		return out
	}

	// If only 1 "sentence" was produced and it's very long, the content likely
	// lacks sentence-ending punctuation (bullet lists, YAML, logs, code).
	// Fall back to splitting on newlines.
	if len(result) == 1 && len([]rune(result[0].text)) > defaultMaxChunkChars {
		fallback := fallbackSplit(text)
		out := make([]sentenceWithRuneSpan, len(fallback))
		for i, s := range fallback {
			out[i] = sentenceWithRuneSpan{text: s}
		}
		return out
	}

	return result
}

// extractSentenceSpans splits a prose fragment (outside code blocks) into
// sentences with rune spans, rebased by offset.
func extractSentenceSpans(prose string, offset int, abbrevs map[string]bool) []sentenceWithRuneSpan {
	var result []sentenceWithRuneSpan
	runes := []rune(prose)
	start := 0

	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if !isSentenceTerminator(r) {
			continue
		}

		if isDecimalPoint(runes, i) || isKnownAbbreviation(runes, i, abbrevs) {
			continue
		}

		if i < len(runes)-1 && !unicode.IsSpace(runes[i+1]) {
			continue
		}

		sent := strings.TrimSpace(string(runes[start : i+1]))
		if sent != "" {
			result = append(result, sentenceWithRuneSpan{
				text:      sent,
				runeStart: start + offset,
				runeEnd:   i + 1 + offset,
			})
		}
		start = i + 1
	}

	// Remaining prose.
	if start < len(runes) {
		tail := strings.TrimSpace(string(runes[start:]))
		if tail != "" {
			result = append(result, sentenceWithRuneSpan{
				text:      tail,
				runeStart: start + offset,
				runeEnd:   len(runes) + offset,
			})
		}
	}

	return result
}

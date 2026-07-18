package chunking

import (
	"strings"
	"unicode/utf8"
)

// LimitOptions configures the hard-limit finalizer.
type LimitOptions struct {
	MaxChars     int
	Overlap      int
	OriginalText string // when set, sub-chunks from splits carry sub-spans mapped back into this text
}

// EnforceHardLimits ensures no chunk exceeds the configured character limit.
// Oversized chunks are split using cascading boundary logic:
//
//	paragraph → sentence → word → hard cut
//
// Undersized chunks are left untouched. Chunk IDs are rebuilt to be sequential.
// Empty chunks are dropped. Text order is preserved.
//
// When an oversized chunk is split, the resulting sub-chunks have StartChar and
// EndChar cleared to 0 because the split text cannot be reliably mapped back to
// the original source byte offsets. Pass-through chunks retain their spans.
func EnforceHardLimits(chunks []Chunk, opts LimitOptions) []Chunk {
	if opts.MaxChars <= 0 {
		return chunks
	}

	var result []Chunk
	id := 0

	for _, c := range chunks {
		if strings.TrimSpace(c.Text) == "" {
			continue
		}

		if utf8.RuneCountInString(c.Text) <= opts.MaxChars {
			c.ID = id
			result = append(result, c)
			id++
			continue
		}

		// Oversized — split it.
		subs := splitOversizedChunk(c.Text, opts.MaxChars)
		if opts.OriginalText != "" && (c.StartChar != 0 || c.EndChar != 0) {
			// Propagate sub-spans by locating each sub-text within the
			// chunk's original byte range.
			origin := opts.OriginalText[c.StartChar:c.EndChar]
			cursor := 0
			for _, sub := range subs {
				sub = strings.TrimSpace(sub)
				if sub == "" {
					continue
				}
				start, end, found := Locate(origin, sub, cursor)
				if found {
					result = append(result, buildChunkWithSpan(id, sub,
						c.StartChar+start,
						c.StartChar+end))
					cursor = end
				} else {
					result = append(result, buildChunk(id, sub))
				}
				id++
			}
		} else {
			for _, sub := range subs {
				sub = strings.TrimSpace(sub)
				if sub == "" {
					continue
				}
				result = append(result, buildChunk(id, sub))
				id++
			}
		}
	}

	return result
}

// splitOversizedChunk splits a single oversized text into pieces that fit
// within maxChars using cascading boundary logic.
func splitOversizedChunk(text string, maxChars int) []string {
	// Try paragraph boundaries first.
	if subs := splitAtBoundary(text, maxChars, splitParagraphs); len(subs) > 1 {
		var result []string
		for _, s := range subs {
			if utf8.RuneCountInString(s) > maxChars {
				result = append(result, splitOversizedChunk(s, maxChars)...)
			} else {
				result = append(result, s)
			}
		}
		return result
	}

	// Try sentence boundaries.
	if subs := splitAtBoundary(text, maxChars, splitSentencesForLimit); len(subs) > 1 {
		var result []string
		for _, s := range subs {
			if utf8.RuneCountInString(s) > maxChars {
				result = append(result, splitOversizedChunk(s, maxChars)...)
			} else {
				result = append(result, s)
			}
		}
		return result
	}

	// Try word boundaries.
	if subs := splitAtBoundary(text, maxChars, splitWordsForLimit); len(subs) > 1 {
		var result []string
		for _, s := range subs {
			if utf8.RuneCountInString(s) > maxChars {
				result = append(result, splitOversizedChunk(s, maxChars)...)
			} else {
				result = append(result, s)
			}
		}
		return result
	}

	// Hard cut fallback — split at maxChars rune boundary.
	return hardCutSplit(text, maxChars)
}

// splitFunc extracts segments from text for boundary-aware splitting.
type splitFunc func(string) []string

// splitAtBoundary attempts to split text using the given boundary function,
// accumulating segments until maxChars would be exceeded.
func splitAtBoundary(text string, maxChars int, fn splitFunc) []string {
	segments := fn(text)
	if len(segments) <= 1 {
		return nil
	}

	var result []string
	var buf strings.Builder
	runeCount := 0

	for _, seg := range segments {
		segRunes := utf8.RuneCountInString(seg)

		if buf.Len() > 0 && runeCount+1+segRunes > maxChars && runeCount >= maxChars/2 {
			// Flush current buffer, but only when we've accumulated at least
			// half the budget to prevent sliver chunks.
			result = append(result, strings.TrimSpace(buf.String()))
			buf.Reset()
			runeCount = 0
		}

		if buf.Len() > 0 {
			buf.WriteString(" ")
			runeCount++
		}
		buf.WriteString(seg)
		runeCount += segRunes
	}

	if buf.Len() > 0 {
		result = append(result, strings.TrimSpace(buf.String()))
	}

	// If we couldn't split into multiple pieces, return nil so the next
	// boundary level is tried.
	if len(result) <= 1 {
		return nil
	}

	return result
}

// splitParagraphs splits text into paragraph-level segments for limiting.
func splitParagraphs(text string) []string {
	return splitIntoParagraphs(text)
}

// splitSentencesForLimit splits text into sentence-level segments for limiting.
func splitSentencesForLimit(text string) []string {
	return splitIntoSentences(text)
}

// splitWordsForLimit splits text into word-level segments for limiting.
func splitWordsForLimit(text string) []string {
	return strings.Fields(text)
}

// hardCutSplit splits text at exact rune boundaries when no other boundary works.
func hardCutSplit(text string, maxChars int) []string {
	runes := []rune(text)
	var result []string

	for i := 0; i < len(runes); {
		end := i + maxChars
		if end > len(runes) {
			end = len(runes)
		}
		result = append(result, string(runes[i:end]))
		i = end
	}

	return result
}

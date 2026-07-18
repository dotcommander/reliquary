package chunking

import (
	"regexp"
	"strings"
)

var (
	conversationTurnRe = regexp.MustCompile(`(?m)^#{1,4}\s+(?:USER|ASSISTANT|SYSTEM|AI)\b`)
	headingRe          = regexp.MustCompile(`(?m)^#{2,3}\s+`)
	horizontalRuleRe   = regexp.MustCompile(`(?m)^[ \t]*[-*_]{3,}[ \t]*$`)
)

// SemanticUnit is a structural unit used by semantic chunk planning.
// StartChar and EndChar are byte offsets into the original input text when
// known. A zero span means the source location is unknown.
type SemanticUnit struct {
	Text      string
	StartChar int
	EndChar   int
}

// SemanticUnits returns the structural semantic units that semantic chunking
// uses before embedding. Units carry source byte offsets when they can be
// mapped exactly.
func SemanticUnits(text string) []SemanticUnit {
	units := semanticUnits(text)
	if len(units) == 0 {
		return nil
	}
	out := make([]SemanticUnit, len(units))
	for i, u := range units {
		out[i] = SemanticUnit{
			Text:      u.text,
			StartChar: u.start,
			EndChar:   u.end,
		}
	}
	return out
}

// semanticUnits attempts to split text into structural semantic units before
// falling back to sentence-level splitting. It tries, in order:
//  1. Conversation turns ("#### USER" markers)
//  2. Level-2+ markdown headings ("## " or "### ")
//  3. Horizontal rules ("---", "***", "___" variants)
//  4. Paragraph blocks (double newlines)
//  5. Sentence fallback
//
// A structural splitter is used only when it produces at least 2 meaningful
// units. Units carry byte-offset spans when possible.
func semanticUnits(text string) []textSpan {
	if text == "" {
		return nil
	}

	// Try conversation turns.
	if units := splitByConversationTurns(text); len(units) >= 2 {
		return units
	}

	// Try markdown headings (## and ###).
	if units := splitByHeadings(text); len(units) >= 2 {
		return units
	}

	// Try horizontal rules.
	if units := splitByHorizontalRules(text); len(units) >= 2 {
		return units
	}

	// Try paragraph blocks.
	if units := splitByParagraphBlocks(text); len(units) >= 2 {
		return units
	}

	// Fall back to sentence-level splitting with spans.
	return splitIntoSentencesWithSpans(text)
}

func semanticUnitsFromPublic(units []SemanticUnit) ([]textSpan, []string) {
	if len(units) == 0 {
		return nil, nil
	}
	spans := make([]textSpan, 0, len(units))
	texts := make([]string, 0, len(units))
	for _, u := range units {
		if strings.TrimSpace(u.Text) == "" {
			continue
		}
		if u.StartChar < 0 || u.EndChar <= u.StartChar {
			u.StartChar = 0
			u.EndChar = 0
		}
		spans = append(spans, textSpan{text: u.Text, start: u.StartChar, end: u.EndChar})
		texts = append(texts, u.Text)
	}
	return spans, texts
}

// splitByConversationTurns splits on "#### USER" (and similar) markers.
// The marker line is excluded from the content of both neighboring units.
func splitByConversationTurns(text string) []textSpan {
	return splitOnPattern(text, conversationTurnRe)
}

// splitByHeadings splits on "## " and "### " markdown headings.
// The heading line is included at the start of the following section.
func splitByHeadings(text string) []textSpan {
	return splitOnHeadingPattern(text, headingRe)
}

// splitByHorizontalRules splits on horizontal rule lines (---, ***, ___ with
// optional spacing). The rule line is excluded from both neighboring units.
func splitByHorizontalRules(text string) []textSpan {
	return splitOnPattern(text, horizontalRuleRe)
}

// splitByParagraphBlocks splits on blank lines (double newlines).
// Each paragraph block becomes one unit.
func splitByParagraphBlocks(text string) []textSpan {
	var units []textSpan
	cursor := 0

	for cursor <= len(text) {
		// Find end of current paragraph block (next blank line).
		end := len(text)
		if idx := strings.Index(text[cursor:], "\n\n"); idx >= 0 {
			end = cursor + idx
		}

		segment := strings.TrimSpace(text[cursor:end])
		if segment != "" {
			// Locate exact byte span for the trimmed segment.
			start, spanEnd, found := Locate(text, segment, cursor)
			if found {
				units = append(units, textSpan{text: segment, start: start, end: spanEnd})
			} else {
				units = append(units, textSpan{text: segment})
			}
		}

		if end == len(text) {
			break
		}
		cursor = end + 2 // skip past "\n\n"
	}

	return units
}

// splitOnPattern is a shared helper for regex-based structural splitting.
// It splits text at each match of re, excluding the matched line from content.
// This produces units from the text between matches.
func splitOnPattern(text string, re *regexp.Regexp) []textSpan {
	locations := re.FindAllStringIndex(text, -1)
	if len(locations) == 0 {
		return nil
	}

	var units []textSpan
	prev := 0

	for _, loc := range locations {
		matchStart := loc[0]

		// Content before this match.
		if matchStart > prev {
			// Find the start of the line containing the match.
			lineStart := matchStart
			if idx := strings.LastIndex(text[prev:matchStart], "\n"); idx >= 0 {
				lineStart = prev + idx + 1
			}
			segment := strings.TrimSpace(text[prev:lineStart])
			if segment != "" {
				start, end, found := Locate(text, segment, prev)
				if found {
					units = append(units, textSpan{text: segment, start: start, end: end})
				} else {
					units = append(units, textSpan{text: segment})
				}
			}
		}

		// Advance past the matched line.
		lineEnd := strings.Index(text[matchStart:], "\n")
		if lineEnd < 0 {
			prev = len(text)
		} else {
			prev = matchStart + lineEnd + 1
		}
	}

	// Trailing content after the last match.
	if prev < len(text) {
		segment := strings.TrimSpace(text[prev:])
		if segment != "" {
			start, end, found := Locate(text, segment, prev)
			if found {
				units = append(units, textSpan{text: segment, start: start, end: end})
			} else {
				units = append(units, textSpan{text: segment})
			}
		}
	}

	return units
}

// splitOnHeadingPattern splits at heading lines while preserving each heading
// as the first line of the following unit.
func splitOnHeadingPattern(text string, re *regexp.Regexp) []textSpan {
	locations := re.FindAllStringIndex(text, -1)
	if len(locations) == 0 {
		return nil
	}

	var units []textSpan

	if first := locations[0][0]; first > 0 {
		segment := strings.TrimSpace(text[:first])
		if segment != "" {
			start, end, found := Locate(text, segment, 0)
			if found {
				units = append(units, textSpan{text: segment, start: start, end: end})
			} else {
				units = append(units, textSpan{text: segment})
			}
		}
	}

	for i, loc := range locations {
		unitStart := loc[0]
		unitEnd := len(text)
		if i+1 < len(locations) {
			unitEnd = locations[i+1][0]
		}
		segment := strings.TrimSpace(text[unitStart:unitEnd])
		if segment == "" {
			continue
		}
		start, end, found := Locate(text, segment, unitStart)
		if found {
			units = append(units, textSpan{text: segment, start: start, end: end})
		} else {
			units = append(units, textSpan{text: segment})
		}
	}

	return units
}

// mergeTinyUnits merges units shorter than minLen into the previous unit.
// Both the textSpan slice and the text slice are kept in sync.
// Spans are cleared when units are merged (combined text no longer maps
// exactly to a single source range).
func mergeTinyUnits(units []textSpan, minLen int) ([]textSpan, []string) {
	if len(units) == 0 {
		return nil, nil
	}
	merged := make([]textSpan, 0, len(units))
	for _, u := range units {
		if len(u.text) < minLen && len(merged) > 0 {
			merged[len(merged)-1].text += " " + u.text
			merged[len(merged)-1].start = 0
			merged[len(merged)-1].end = 0
		} else {
			merged = append(merged, u)
		}
	}
	texts := make([]string, len(merged))
	for i, u := range merged {
		texts[i] = u.text
	}
	return merged, texts
}

// buildSemanticChunks creates Chunk slices from grouped text, attempting to
// recover byte spans from the original semantic units. When a group's text
// matches a contiguous span of source units, the span is preserved; otherwise
// it is cleared to 0,0 (unknown).
func buildSemanticChunks(units []textSpan, groups []string, originalText string) []Chunk {
	chunks := make([]Chunk, len(groups))
	unitIdx := 0
	for i, g := range groups {
		if unitIdx >= len(units) {
			chunks[i] = buildChunk(i, g)
			continue
		}
		groupText := g
		firstUnit := unitIdx
		consumed := 0
		for unitIdx < len(units) {
			u := units[unitIdx]
			if len(groupText) < len(u.text) {
				break
			}
			if strings.HasPrefix(groupText, u.text) {
				groupText = strings.TrimSpace(groupText[len(u.text):])
				unitIdx++
				consumed++
				continue
			}
			if consumed > 0 && strings.HasPrefix(groupText, " "+u.text) {
				groupText = groupText[len(u.text)+1:]
				unitIdx++
				consumed++
				continue
			}
			break
		}
		if consumed == 0 {
			chunks[i] = buildChunk(i, g)
			continue
		}
		first := units[firstUnit]
		last := units[unitIdx-1]
		if first.start != 0 || first.end != 0 {
			spanStart := first.start
			spanEnd := last.end
			if spanStart < len(originalText) && spanEnd <= len(originalText) && spanStart < spanEnd {
				src := originalText[spanStart:spanEnd]
				cleaned := strings.TrimSpace(src)
				if cleaned == g {
					chunks[i] = buildChunkWithSpan(i, g, spanStart, spanEnd)
					continue
				}
			}
		}
		chunks[i] = buildChunk(i, g)
	}
	return chunks
}

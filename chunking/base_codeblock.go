package chunking

import "unicode"

// codeBlockRange represents a protected rune range that should not be split
// by the punctuation scanner. Used for fenced and indented code blocks.
type codeBlockRange struct {
	runeStart int // inclusive rune index
	runeEnd   int // exclusive rune index
}

// findCodeBlockRanges scans text in rune space and returns the protected
// ranges covering fenced code blocks (```...```) and indented code blocks
// (4+ spaces or tab at line start). Fenced blocks include the opener and
// closer. Unclosed fenced blocks run from opener to end of text.
func findCodeBlockRanges(runes []rune) []codeBlockRange {
	var ranges []codeBlockRange
	i := 0
	for i < len(runes) {
		// Skip to the start of a line.
		if i > 0 && runes[i-1] != '\n' {
			i++
			continue
		}

		// Check for fenced code block opener.
		if isFencedOpener(runes, i) {
			end := findFencedCloser(runes, i)
			ranges = append(ranges, codeBlockRange{runeStart: i, runeEnd: end})
			i = end
			continue
		}

		// Check for indented code block (4 spaces or tab).
		if isIndentedBlockStart(runes, i) {
			end := findIndentedBlockEnd(runes, i)
			ranges = append(ranges, codeBlockRange{runeStart: i, runeEnd: end})
			i = end
			continue
		}

		i++
	}
	return ranges
}

// isFencedOpener returns true if runes[pos:] starts with ``` (optionally preceded
// by whitespace) at the beginning of a line.
func isFencedOpener(runes []rune, pos int) bool {
	j := pos
	// Skip optional leading whitespace.
	for j < len(runes) && (runes[j] == ' ' || runes[j] == '\t') {
		j++
	}
	return j+2 < len(runes) && runes[j] == '`' && runes[j+1] == '`' && runes[j+2] == '`'
}

// findFencedCloser scans forward from the fenced opener at pos to find the
// closing ``` line. Returns the rune index after the closing line (or end of
// text if unclosed).
func findFencedCloser(runes []rune, pos int) int {
	// Skip past the opener line.
	pos = indexOfNewline(runes, pos)
	if pos < len(runes) {
		pos++ // past the \n
	}
	for pos < len(runes) {
		j := pos
		for j < len(runes) && (runes[j] == ' ' || runes[j] == '\t') {
			j++
		}
		if j+2 < len(runes) && runes[j] == '`' && runes[j+1] == '`' && runes[j+2] == '`' {
			// Found closer — advance past this line.
			pos = indexOfNewline(runes, pos)
			if pos < len(runes) {
				pos++
			}
			return pos
		}
		pos = indexOfNewline(runes, pos)
		if pos < len(runes) {
			pos++
		}
	}
	return len(runes)
}

// isIndentedBlockStart returns true if the line at pos starts with 4 spaces or
// a tab character.
func isIndentedBlockStart(runes []rune, pos int) bool {
	if pos >= len(runes) {
		return false
	}
	// 4 spaces.
	if pos+3 < len(runes) && runes[pos] == ' ' && runes[pos+1] == ' ' && runes[pos+2] == ' ' && runes[pos+3] == ' ' {
		return true
	}
	// Tab.
	return runes[pos] == '\t'
}

// findIndentedBlockEnd scans forward collecting consecutive indented lines.
// A blank line is included only if the following non-blank line is also
// indented. Otherwise the block ends before the blank line.
func findIndentedBlockEnd(runes []rune, pos int) int {
	for pos < len(runes) {
		if !isIndentedBlockStart(runes, pos) {
			// Blank line: peek ahead to see if the next non-blank line
			// is also indented.
			if isBlankLine(runes, pos) {
				peek := pos
				for peek < len(runes) && runes[peek] == '\n' {
					peek++
				}
				if peek < len(runes) && isIndentedBlockStart(runes, peek) {
					pos = peek
					continue
				}
			}
			break
		}
		pos = indexOfNewline(runes, pos)
		if pos < len(runes) {
			pos++
		}
	}
	return pos
}

// isBlankLine returns true if the line starting at pos contains only whitespace.
func isBlankLine(runes []rune, pos int) bool {
	for pos < len(runes) && runes[pos] != '\n' {
		if !unicode.IsSpace(runes[pos]) {
			return false
		}
		pos++
	}
	return true
}

// indexOfNewline returns the index of the next \n from pos, or len(runes).
func indexOfNewline(runes []rune, pos int) int {
	for pos < len(runes) {
		if runes[pos] == '\n' {
			return pos
		}
		pos++
	}
	return len(runes)
}

// isInsideCodeBlock returns the codeBlockRange containing rune index i, or nil.
func isInsideCodeBlock(ranges []codeBlockRange, i int) *codeBlockRange {
	for idx := range ranges {
		if i >= ranges[idx].runeStart && i < ranges[idx].runeEnd {
			return &ranges[idx]
		}
	}
	return nil
}

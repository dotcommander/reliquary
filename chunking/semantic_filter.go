package chunking

import "strings"

// markAnalyzableUnits returns a bool slice indicating which units should be
// embedded for semantic comparison. Code fences and table blocks are marked
// non-analyzable; plain prose is analyzable.
func markAnalyzableUnits(texts []string) []bool {
	analyzable := make([]bool, len(texts))
	for i, t := range texts {
		analyzable[i] = isAnalyzableText(t)
	}
	return analyzable
}

// isAnalyzableText returns false for text that appears to be a fenced code
// block or a markdown table — these should not drive embedding breakpoints.
func isAnalyzableText(text string) bool {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return true
	}

	// Check for fenced code block.
	first := strings.TrimSpace(lines[0])
	if strings.HasPrefix(first, "```") {
		return false
	}

	// Check for table block (starts with | and has |---| separator).
	if strings.HasPrefix(first, "|") && len(lines) > 1 {
		for _, line := range lines[1:] {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "|") && strings.Contains(trimmed, "---") {
				return false
			}
		}
	}

	return true
}

// filterAnalyzableTexts returns only the texts marked as analyzable.
func filterAnalyzableTexts(texts []string, analyzable []bool) []string {
	var out []string
	for i, a := range analyzable {
		if a {
			out = append(out, texts[i])
		}
	}
	return out
}

// mapBreaksToFullUnits converts break indices from the analyzable-only subsequence
// back to full unit indices. Non-analyzable units are never break points; they
// attach to the preceding chunk.
func mapBreaksToFullUnits(analyzableBreaks []int, analyzable []bool) []int {
	if len(analyzableBreaks) == 0 {
		return nil
	}

	// Build mapping from analyzable index to full unit index.
	analyzableToFull := make([]int, 0, len(analyzable))
	for i, a := range analyzable {
		if a {
			analyzableToFull = append(analyzableToFull, i)
		}
	}

	var result []int
	for _, b := range analyzableBreaks {
		if b+1 < len(analyzableToFull) {
			// Break after analyzable unit b means split before full unit analyzableToFull[b+1].
			fullIdx := analyzableToFull[b+1]
			result = append(result, fullIdx-1)
		} else if b < len(analyzableToFull) {
			// Last analyzable unit.
			fullIdx := analyzableToFull[b]
			result = append(result, fullIdx-1)
		}
	}
	return result
}

package chunking

import "strings"

// LineForOffset returns the 1-based line number for the given byte offset in
// content. offset is clamped to [0, len(content)]. An empty content returns 1.
func LineForOffset(content string, offset int) int {
	if offset <= 0 {
		return 1
	}
	if offset > len(content) {
		offset = len(content)
	}
	return strings.Count(content[:offset], "\n") + 1
}

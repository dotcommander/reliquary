package chunking

// ChunkSpan represents a resolved byte range [Start, End) within source text.
type ChunkSpan struct {
	Start int
	End   int
}

// ResolveChunkSpan determines the byte span of chunk.Text within content.
//
// If chunk.StartChar and chunk.EndChar describe a valid byte range and
// content[start:end] == chunk.Text, that range is returned directly.
// Otherwise, Locate is used to find the fragment starting from cursor.
// Returns false if the chunk text is empty or cannot be located.
func ResolveChunkSpan(content string, chunk Chunk, cursor int) (ChunkSpan, bool) {
	if chunk.Text == "" {
		return ChunkSpan{}, false
	}

	// Validate existing span: both start and end must be in bounds, and
	// the slice must match the chunk text verbatim. A chunk at the very
	// beginning of content can legitimately have StartChar == 0.
	if chunk.StartChar >= 0 && chunk.EndChar > chunk.StartChar && chunk.EndChar <= len(content) {
		if content[chunk.StartChar:chunk.EndChar] == chunk.Text {
			return ChunkSpan{Start: chunk.StartChar, End: chunk.EndChar}, true
		}
	}

	// Normalize cursor.
	if cursor < 0 {
		cursor = 0
	}

	start, end, ok := Locate(content, chunk.Text, cursor)
	if !ok {
		return ChunkSpan{}, false
	}
	return ChunkSpan{Start: start, End: end}, true
}

// NextChunkCursor returns the next search cursor after a resolved span.
// It advances to Start+1 (not End) so that overlapping chunks and repeated
// phrases can still be found while avoiding the same match.
func NextChunkCursor(span ChunkSpan) int {
	if span.End <= span.Start {
		return span.Start
	}
	return span.Start + 1
}

// LineRangeForSpan returns the inclusive 1-based line range for a byte span
// within content. The end offset is adjusted by -1 before line counting so
// that a span ending exactly at a newline reports the previous content line
// as the end line, not the next.
func LineRangeForSpan(content string, span ChunkSpan) (startLine int, endLine int) {
	startLine = LineForOffset(content, span.Start)
	endOffset := span.End
	if span.End > span.Start {
		endOffset = span.End - 1
	}
	return startLine, LineForOffset(content, endOffset)
}

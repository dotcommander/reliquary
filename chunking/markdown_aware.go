package chunking

import (
	"regexp"
	"strings"
	"unicode/utf8"
)

// markdownAwareChunker splits text respecting markdown structure:
// code fences, headings, and paragraphs are kept intact when possible.
type markdownAwareChunker struct{}

func newMarkdownAwareChunker() *markdownAwareChunker {
	return &markdownAwareChunker{}
}

func (m *markdownAwareChunker) Strategy() Strategy {
	return MarkdownAware
}

// separatorRowRe matches a markdown table separator row: cells containing only
// -, :, whitespace, and optional pipes. Must have at least one dash per cell.
var separatorRowRe = regexp.MustCompile(`^\|?[\s:]*-[-\s:]*\|.+\|?[\s:]*$`)

func (m *markdownAwareChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}

	mdBlocks := extractMarkdownBlocks([]byte(text))
	if len(mdBlocks) == 0 {
		return nil
	}

	var chunks []Chunk
	chunkID := 0
	current := ""
	currentStart, currentEnd := 0, 0
	currentMeta := map[string]string(nil) // metadata for single-block accumulation
	singleBlock := true                   // whether current accumulation is a single block

	for _, blk := range mdBlocks {
		block := blk.text
		candidate := block
		if current != "" {
			candidate = current + "\n\n" + block
		}

		if utf8.RuneCountInString(candidate) <= size {
			current = candidate
			if current == block {
				currentStart = blk.startByte
				currentEnd = blk.endByte
				currentMeta = blk.metadata
			} else {
				currentEnd = blk.endByte
				currentMeta = nil // merged blocks lose metadata
				singleBlock = false
			}
			continue
		}

		// Emit the accumulated block.
		if current != "" {
			prevLen := len(chunks)
			chunks = appendChunkIfValid(chunks, chunkID, current, text, currentStart, currentEnd)
			if len(chunks) > prevLen {
				if singleBlock {
					chunks[len(chunks)-1].Metadata = currentMeta
				}
				chunkID++
			}
		}

		// If the single block fits, start a new accumulation.
		if utf8.RuneCountInString(block) <= size {
			current = block
			currentStart = blk.startByte
			currentEnd = blk.endByte
			currentMeta = blk.metadata
			singleBlock = true
			continue
		}

		// Block exceeds size — try table-aware splitting first.
		if tableParts, ok := splitMarkdownTableBlock(block, size); ok {
			for _, tp := range tableParts {
				chunks = append(chunks, buildChunk(chunkID, tp))
				chunkID++
			}
			current = ""
			currentStart = 0
			currentEnd = 0
			currentMeta = nil
			singleBlock = true
			continue
		}

		// Block exceeds size — fall back to word-boundary splitting.
		wordChunker := newWordBoundaryChunker()
		sub := wordChunker.Chunk(block, size, overlap)
		// Rebase sub-chunk spans from block-relative to original-text-relative.
		sub = adjustChunkSpans(sub, blk.startByte)
		for _, sc := range sub {
			sc.ID = chunkID
			chunks = append(chunks, sc)
			chunkID++
		}
		current = ""
		currentStart = 0
		currentEnd = 0
		currentMeta = nil
		singleBlock = true
	}

	if strings.TrimSpace(current) != "" {
		prevLen := len(chunks)
		chunks = appendChunkIfValid(chunks, chunkID, current, text, currentStart, currentEnd)
		if len(chunks) > prevLen {
			if singleBlock {
				chunks[len(chunks)-1].Metadata = currentMeta
			}
		}
	}

	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size, Overlap: overlap, OriginalText: text})
}

// splitMarkdownTableBlock detects whether block is a pipe-delimited table
// (either raw markdown with a separator row, or goldmark-reformatted rows)
// and splits it into chunks that preserve header context.
// Returns (parts, true) if the block is a valid table, or ([], false) otherwise.
//
// Each emitted chunk includes the header row (and separator row if present in
// source) followed by as many body rows as fit under size. Body row order is
// preserved. No rows are dropped.
//
// Because later chunks duplicate the header, spans are not returned; callers
// should use buildChunk (zero spans) for these table chunks.
func splitMarkdownTableBlock(block string, size int) ([]string, bool) {
	lines := strings.Split(block, "\n")

	// Collect non-empty lines.
	var nonEmpty []string
	for _, l := range lines {
		trimmed := strings.TrimRight(l, " \t\r")
		if trimmed != "" {
			nonEmpty = append(nonEmpty, trimmed)
		}
	}

	// A table needs at least 2 non-empty lines: header + at least one body row
	// (or header + separator for raw markdown).
	if len(nonEmpty) < 2 {
		return nil, false
	}

	// All non-empty lines must contain pipes.
	for _, l := range nonEmpty {
		if !strings.Contains(l, "|") {
			return nil, false
		}
	}

	header := nonEmpty[0]
	separator := ""
	bodyStart := 1

	// Check if the second line is a markdown separator row.
	if len(nonEmpty) > 1 && separatorRowRe.MatchString(nonEmpty[1]) {
		separator = nonEmpty[1]
		bodyStart = 2
	}

	if bodyStart >= len(nonEmpty) {
		// Header (+ separator) only — emit as one chunk.
		if separator != "" {
			return []string{header + "\n" + separator}, true
		}
		return []string{header}, true
	}

	// Build the context prefix (header + optional separator).
	prefix := header
	if separator != "" {
		prefix = header + "\n" + separator
	}
	prefixLen := utf8.RuneCountInString(prefix)

	bodyLines := nonEmpty[bodyStart:]

	var parts []string
	var buf strings.Builder
	buf.WriteString(prefix)
	bufLen := prefixLen

	for _, row := range bodyLines {
		rowLen := utf8.RuneCountInString(row)
		newLen := bufLen + 1 + rowLen // +1 for newline

		if newLen > size {
			// Flush current chunk if it has body rows.
			if bufLen > prefixLen {
				parts = append(parts, buf.String())
				buf.Reset()
				buf.WriteString(prefix)
				bufLen = prefixLen
			}

			// If a single row + prefix exceeds size, emit it anyway
			// (EnforceHardLimits will apply the final fallback).
			singleRow := prefix + "\n" + row
			if utf8.RuneCountInString(singleRow) > size {
				parts = append(parts, singleRow)
				continue
			}
		}

		buf.WriteByte('\n')
		buf.WriteString(row)
		bufLen += 1 + rowLen
	}

	// Flush remaining.
	if buf.Len() > 0 {
		remaining := buf.String()
		// Emit if there are body rows beyond the prefix, or this is the only chunk.
		newlineCount := strings.Count(remaining, "\n")
		prefixNewlines := strings.Count(prefix, "\n")
		if newlineCount > prefixNewlines || (bufLen == prefixLen && len(parts) == 0) {
			parts = append(parts, remaining)
		}
	}

	if len(parts) == 0 {
		return nil, false
	}

	return parts, true
}

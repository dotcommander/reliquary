package chunking

import (
	"maps"
	"regexp"
	"strconv"
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
	current := ""
	currentStart, currentEnd := 0, 0
	currentType := ""
	var currentMeta map[string]string

	flush := func() {
		previous := len(chunks)
		chunks = appendChunkIfValid(chunks, previous, current, text, currentStart, currentEnd)
		if len(chunks) > previous {
			chunks[len(chunks)-1].Metadata = currentMeta
		}
		current = ""
		currentStart, currentEnd = 0, 0
		currentType = ""
		currentMeta = nil
	}

	for _, blk := range mdBlocks {
		// Keep unlike structural blocks separate. Code and table blocks are
		// also kept individually so their metadata never describes prose.
		if current != "" && (currentType != blk.blockType || blk.blockType == "code" || blk.blockType == "table") {
			flush()
		}
		candidate := blk.text
		if current != "" {
			candidate = current + "\n\n" + blk.text
		}
		if utf8.RuneCountInString(candidate) <= size {
			if current == "" {
				currentStart = blk.startByte
				currentMeta = maps.Clone(blk.metadata)
				currentType = blk.blockType
			} else {
				currentMeta[metaKeyWordCount] = strconv.Itoa(len(strings.Fields(candidate)))
			}
			current, currentEnd = candidate, blk.endByte
			continue
		}

		flush()
		if utf8.RuneCountInString(blk.text) <= size {
			current = blk.text
			currentStart, currentEnd = blk.startByte, blk.endByte
			currentType = blk.blockType
			currentMeta = maps.Clone(blk.metadata)
			continue
		}

		if blk.blockType == "table" {
			if parts, ok := splitMarkdownTableBlock(blk.text, size); ok {
				for _, part := range parts {
					chunk := buildChunk(len(chunks), part)
					chunk.Metadata = maps.Clone(blk.metadata)
					chunks = append(chunks, chunk)
				}
				continue
			}
		}

		sub := newWordBoundaryChunker().Chunk(blk.text, size, overlap)
		// Rebase only when the block itself is verbatim. Reconstructed blocks
		// cannot provide trustworthy source provenance for their fragments.
		verbatim := blk.startByte >= 0 && blk.endByte > blk.startByte && blk.endByte <= len(text) && text[blk.startByte:blk.endByte] == blk.text
		for _, chunk := range sub {
			chunk.ID = len(chunks)
			chunk.Metadata = maps.Clone(blk.metadata)
			if verbatim && chunk.EndChar > chunk.StartChar {
				chunk.StartChar += blk.startByte
				chunk.EndChar += blk.startByte
			} else {
				chunk.StartChar, chunk.EndChar = 0, 0
			}
			chunks = append(chunks, chunk)
		}
	}
	flush()

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

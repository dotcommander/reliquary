package chunking

import (
	"strings"
	"unicode/utf8"
)

// headingAwareChunker recursively splits markdown by heading level.
// When a section exceeds the size limit, it subdivides by the next heading
// level (H1→H2→H3→...) before falling back to sentence-boundary splitting.
type headingAwareChunker struct {
	sentenceFallback *smartBoundaryChunker
}

func newHeadingAwareChunker() *headingAwareChunker {
	return &headingAwareChunker{
		sentenceFallback: newSmartBoundaryChunker(),
	}
}

func (h *headingAwareChunker) Strategy() Strategy {
	return HeadingAware
}

// section is a heading-delimited block of text with its heading level.
type section struct {
	level   int // 0 = preamble (text before any heading)
	heading string
	content string
}

func (h *headingAwareChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}

	results := h.splitWithPath(text, 1, size, nil)

	var chunks []Chunk
	id := 0
	cursor := 0
	for _, r := range results {
		trimmed := strings.TrimSpace(r.text)
		if trimmed == "" {
			continue
		}

		if utf8.RuneCountInString(trimmed) <= size {
			startChar, endChar, _ := Locate(text, trimmed, cursor)
			cursor = endChar
			prevLen := len(chunks)
			chunks = appendChunkIfValid(chunks, id, trimmed, "", startChar, endChar)
			if len(chunks) > prevLen {
				chunks[len(chunks)-1].Path = r.path
				id++
			}
			continue
		}

		sectionStart, _, _ := Locate(text, trimmed, cursor)
		sub := h.sentenceFallback.Chunk(trimmed, size, overlap)
		sub = adjustChunkSpans(sub, sectionStart)
		for _, sc := range sub {
			sc.ID = id
			sc.Path = r.path
			chunks = append(chunks, sc)
			id++
		}
	}

	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size, Overlap: overlap, OriginalText: text})
}

// splitRecursive splits text by headings at minLevel, then recursively
// subdivides oversized sections by the next heading level.
// pathSection is a section string with its accumulated heading breadcrumb.
type pathSection struct {
	text string
	path []string
}

// splitWithPath splits recursively, threading heading breadcrumbs through.
func (h *headingAwareChunker) splitWithPath(text string, minLevel, size int, parentPath []string) []pathSection {
	if minLevel > 6 {
		return []pathSection{{text: text, path: copySlice(parentPath)}}
	}

	sections := splitByLevel(text, minLevel)

	// If only one section found (no split at this level), try next level.
	if len(sections) <= 1 {
		// If there's a single section with a heading, include it in the path.
		if len(sections) == 1 && sections[0].level > 0 {
			headingText := strings.TrimLeft(sections[0].heading, "# ")
			newPath := updateSectionPath(copySlice(parentPath), headingText, sections[0].level)
			return h.splitWithPath(text, minLevel+1, size, newPath)
		}
		return h.splitWithPath(text, minLevel+1, size, parentPath)
	}

	var result []pathSection
	for _, sec := range sections {
		full := sec.fullText()
		if full == "" {
			continue
		}

		// Build this section's path from parent + this heading.
		var secPath []string
		if sec.level > 0 {
			headingText := strings.TrimLeft(sec.heading, "# ")
			secPath = updateSectionPath(copySlice(parentPath), headingText, sec.level)
		} else {
			secPath = copySlice(parentPath)
		}

		if utf8.RuneCountInString(full) <= size {
			result = append(result, pathSection{text: full, path: secPath})
			continue
		}

		// Oversized — recurse into next heading level within this section's content.
		if sec.level > 0 && sec.level < 6 {
			sub := h.splitWithPath(sec.content, sec.level+1, size, secPath)
			for i, s := range sub {
				if i == 0 && sec.heading != "" {
					s.text = sec.heading + "\n" + s.text
				}
				result = append(result, s)
			}
		} else {
			result = append(result, pathSection{text: full, path: secPath})
		}
	}

	return result
}

// updateSectionPath trims the path to the appropriate level and appends the heading.
// Level 0 (preamble) does not modify the path.
func updateSectionPath(path []string, heading string, level int) []string {
	if level <= 0 || heading == "" {
		return path
	}
	if level <= len(path) {
		path = path[:level-1]
	}
	return append(path, heading)
}

func copySlice(original []string) []string {
	if original == nil {
		return nil
	}
	copied := make([]string, len(original))
	copy(copied, original)
	return copied
}

func (s section) fullText() string {
	full := s.heading
	if s.content == "" {
		return strings.TrimSpace(full)
	}
	if full != "" {
		full += "\n" + s.content
	} else {
		full = s.content
	}
	return strings.TrimSpace(full)
}

// splitByLevel splits text into sections at headings of exactly the given level.
// Uses goldmark for structural parsing — headings inside code fences are never
// detected because goldmark separates them by node type.
// Text before the first heading becomes the preamble (level 0).
func splitByLevel(text string, level int) []section {
	blocks := extractMarkdownBlocks([]byte(text))
	if len(blocks) == 0 {
		trimmed := strings.TrimSpace(text)
		if trimmed == "" {
			return nil
		}
		return []section{{level: 0, content: trimmed}}
	}

	var sections []section
	var cur section
	var contentParts []string

	flush := func() {
		cur.content = strings.TrimSpace(strings.Join(contentParts, "\n"))
		contentParts = nil
		cur.heading = strings.TrimSpace(cur.heading)
		if cur.heading != "" || cur.content != "" {
			sections = append(sections, cur)
		}
		cur = section{}
	}

	for _, blk := range blocks {
		if blk.blockType == "heading" && blk.level == level {
			flush()
			cur.level = level
			cur.heading = blk.text
			continue
		}
		contentParts = append(contentParts, blk.text)
	}

	flush()
	return sections
}

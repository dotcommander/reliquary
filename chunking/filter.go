package chunking

import (
	"strconv"
)

// ProseWordFloor is the minimum word count for any block type to be admitted
// by FilterProse.
const ProseWordFloor = 5

// HeadingWordFloor is the minimum word count for a heading block to be
// admitted by FilterProse (lower than ProseWordFloor because headings are
// typically short but still meaningful).
const HeadingWordFloor = 3

// FilterProse returns only prose-bearing chunks, skipping code and tables,
// and dropping chunks below word-count floors. It reads Chunk.Metadata keys
// "type" and "word_count". Chunks without Metadata are passed through
// unchanged — they cannot be classified, so FilterProse does not drop them.
func FilterProse(chunks []Chunk) []Chunk {
	if len(chunks) == 0 {
		return nil
	}
	out := make([]Chunk, 0, len(chunks))
	for _, c := range chunks {
		if admitChunk(c) {
			out = append(out, c)
		}
	}
	return out
}

// admitChunk returns true if the chunk should be included in prose output.
func admitChunk(c Chunk) bool {
	if c.Metadata == nil {
		return true // cannot classify — pass through
	}

	blockType := c.Metadata["type"]

	// Skip code blocks and tables entirely.
	if blockType == "code" || blockType == "table" {
		return false
	}

	// Parse word count from metadata.
	wcStr := c.Metadata[metaKeyWordCount]
	if wcStr == "" {
		// For headings, word_count may not be set; count from text.
		if blockType == "heading" {
			// Heading text includes # prefix; the word count is from metadata.
			// If no word_count metadata, admit by default.
			return true
		}
		return true // unknown word count — admit
	}
	wc, err := strconv.Atoi(wcStr)
	if err != nil {
		return true // unparseable — admit
	}

	// Apply word-count floors by block type.
	switch blockType {
	case "heading":
		return wc >= HeadingWordFloor
	default:
		return wc >= ProseWordFloor
	}
}

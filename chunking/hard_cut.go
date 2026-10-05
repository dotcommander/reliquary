package chunking

// hardCutChunker splits text into fixed-size rune slices
// with no boundary awareness.
type hardCutChunker struct{}

func newHardCutChunker() *hardCutChunker {
	return &hardCutChunker{}
}

func (h *hardCutChunker) Strategy() Strategy {
	return HardCut
}

func (h *hardCutChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}

	if overlap < 0 {
		overlap = 0
	}
	if overlap >= size {
		overlap = size - 1
	}

	var chunks []Chunk
	runes := []rune(text)
	chunkID := 0

	// Build rune-index → byte-offset mapping for source spans.
	byteOffsets := runeByteOffsets(text, len(runes))

	for i := 0; i < len(runes); {
		end := i + size
		if end > len(runes) {
			end = len(runes)
		}

		start := i

		chunkText := string(runes[start:end])
		spanStart := byteOffsets[start]
		spanEnd := byteOffsets[end]
		chunks = append(chunks, buildChunkWithSpan(chunkID, chunkText, spanStart, spanEnd))

		if end == len(runes) {
			break
		}
		i = end - overlap
		chunkID++
	}

	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size, Overlap: overlap, OriginalText: text})
}

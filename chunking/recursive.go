package chunking

import (
	"unicode"
)

// recursiveChunker splits text by recursively applying an ordered separator
// profile from broad to narrow, following the semantics popularized by
// LangChain's RecursiveCharacterTextSplitter: split by the first separator
// that appears, recurse into each oversized piece with the remaining
// separators, then greedily merge adjacent pieces while they fit the size
// budget. Merged chunk text is the verbatim source slice (separators between
// merged pieces included), so StartChar/EndChar map exactly into the input.
type recursiveChunker struct {
	profile SeparatorProfile
}

func newRecursiveChunker() *recursiveChunker {
	return &recursiveChunker{profile: DefaultTextSeparatorProfile()}
}

// NewRecursiveChunker returns a recursive Chunker driven by the given
// separator profile, ordered broad to narrow. An empty profile defaults to
// DefaultTextSeparatorProfile. A trailing "" separator enables rune-level
// hard cuts for pieces no other separator can split; without it, unbreakable
// oversized pieces are emitted whole and left to EnforceHardLimits.
func NewRecursiveChunker(profile SeparatorProfile) Chunker {
	if len(profile.Separators) == 0 {
		profile = DefaultTextSeparatorProfile()
	}
	return &recursiveChunker{profile: profile}
}

func (c *recursiveChunker) Strategy() Strategy {
	return Recursive
}

// Chunk splits text into chunks of at most size runes (except unbreakable
// oversized pieces, which EnforceHardLimits then cascades). overlap is the
// number of trailing runes of the previous chunk prepended to the next; a
// value of size or more is clamped so the chunker always makes progress.
func (c *recursiveChunker) Chunk(text string, size int, overlap int) []Chunk {
	if size <= 0 || text == "" {
		return nil
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= size {
		overlap = size - 1
	}

	runes := []rune(text)
	separators := c.profile.SeparatorStrings()

	leaves := recursiveSplitRunes(runes, 0, len(runes), size, separators)
	chunks := mergeRuneSpans(text, runes, leaves, size, overlap)

	return EnforceHardLimits(chunks, LimitOptions{MaxChars: size, Overlap: overlap, OriginalText: text})
}

// runeSpan is a [start, end) rune-index range into the source text.
type runeSpan struct {
	start int
	end   int
}

// recursiveSplitRunes splits runes[start:end] into leaf spans. Pieces that
// fit the size budget become leaves immediately; oversized pieces recurse
// with the remaining separators. Recursion depth is bounded by the number of
// separators because each level consumes at least one.
func recursiveSplitRunes(runes []rune, start, end, size int, separators []string) []runeSpan {
	length := end - start
	if length <= 0 {
		return nil
	}
	if length <= size {
		return []runeSpan{{start: start, end: end}}
	}

	for i, sep := range separators {
		if sep == "" {
			// Rune-level hard cut: nothing narrower remains.
			return hardCutRuneSpans(start, end, size)
		}
		pieces := splitAtRuneSeparator(runes, start, end, sep)
		if len(pieces) <= 1 {
			continue
		}
		var spans []runeSpan
		for _, piece := range pieces {
			spans = append(spans, recursiveSplitRunes(runes, piece.start, piece.end, size, separators[i+1:])...)
		}
		return spans
	}

	// No separator matches and the piece is oversized: keep it whole and let
	// EnforceHardLimits apply the paragraph→sentence→word→hard-cut cascade.
	return []runeSpan{{start: start, end: end}}
}

// hardCutRuneSpans returns consecutive [start, end) spans of at most size
// runes each.
func hardCutRuneSpans(start, end, size int) []runeSpan {
	var spans []runeSpan
	for i := start; i < end; i += size {
		cut := i + size
		if cut > end {
			cut = end
		}
		spans = append(spans, runeSpan{start: i, end: cut})
	}
	return spans
}

// splitAtRuneSeparator splits runes[start:end] at each occurrence of sep,
// returning the pieces between separators. The separator bytes themselves are
// excluded from pieces; when merging, they are covered by the contiguous span
// from the first to the last merged piece. Returns a single span when sep
// does not occur.
func splitAtRuneSeparator(runes []rune, start, end int, sep string) []runeSpan {
	sepRunes := []rune(sep)
	n := len(sepRunes)
	var spans []runeSpan
	pieceStart := start
	for i := start; i < end; {
		if i+n <= end && runeSliceEqual(runes[i:i+n], sepRunes) {
			spans = append(spans, runeSpan{start: pieceStart, end: i})
			i += n
			pieceStart = i
			continue
		}
		i++
	}
	spans = append(spans, runeSpan{start: pieceStart, end: end})
	return spans
}

func runeSliceEqual(a, b []rune) bool {
	for i, r := range a {
		if r != b[i] {
			return false
		}
	}
	return true
}

// mergeRuneSpans greedily merges adjacent leaf spans into chunks of at most
// size runes, prepending the previous chunk's last overlap runes to each
// chunk after the first. Leading and trailing whitespace is trimmed in rune
// space so every emitted span satisfies source[StartChar:EndChar] == Text.
func mergeRuneSpans(source string, runes []rune, leaves []runeSpan, size, overlap int) []Chunk {
	off := runeByteOffsets(source, len(runes))

	var chunks []Chunk
	id := 0
	emit := func(start, end int) {
		for start < end && unicode.IsSpace(runes[start]) {
			start++
		}
		for end > start && unicode.IsSpace(runes[end-1]) {
			end--
		}
		if start >= end {
			return
		}
		chunks = append(chunks, buildChunkWithSpan(id, string(runes[start:end]), off[start], off[end]))
		id++
	}

	chunkStart, chunkEnd := -1, -1
	for _, leaf := range leaves {
		if leaf.start >= leaf.end {
			continue
		}
		if chunkStart < 0 {
			chunkStart, chunkEnd = leaf.start, leaf.end
			continue
		}
		if leaf.end-chunkStart <= size {
			chunkEnd = leaf.end
			continue
		}

		// The leaf does not fit: flush the current chunk and restart inside
		// its trailing overlap window.
		emit(chunkStart, chunkEnd)
		next := leaf.start
		if overlap > 0 {
			next = chunkEnd - overlap
		}
		if next <= chunkStart {
			next = chunkStart + 1
		}
		chunkStart = next

		// The leaf alone may still overflow from the overlapped start.
		for leaf.end-chunkStart > size {
			cut := chunkStart + size
			emit(chunkStart, cut)
			chunkStart = cut - overlap
			if chunkStart <= cut-size {
				chunkStart = cut - size + 1
			}
		}
		chunkEnd = leaf.end
	}
	if chunkStart >= 0 && chunkEnd > chunkStart {
		emit(chunkStart, chunkEnd)
	}
	return chunks
}

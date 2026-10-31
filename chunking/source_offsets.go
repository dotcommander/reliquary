package chunking

import (
	"sort"
	"strings"
	"unicode/utf8"
)

// SourceOffsets indexes one source for repeated byte-to-line and byte-to-rune
// projections. Sparse checkpoints bound rescanning without retaining one index
// entry per byte, rune, or newline. It is immutable after construction.
type SourceOffsets struct {
	source      string
	checkpoints []sourceCheckpoint
}

type sourceCheckpoint struct {
	byteOffset int
	runes      int
	newlines   int
}

// Checkpoints land on rune boundaries, at most three bytes past this spacing.
const sourceCheckpointSpacing = 256

// NewSourceOffsets scans source once for reusable line and rune checkpoints.
func NewSourceOffsets(source string) *SourceOffsets {
	index := &SourceOffsets{source: source, checkpoints: []sourceCheckpoint{{}}}
	var runes, newlines int
	for pos, r := range source {
		last := index.checkpoints[len(index.checkpoints)-1]
		if pos-last.byteOffset >= sourceCheckpointSpacing {
			index.checkpoints = append(index.checkpoints, sourceCheckpoint{pos, runes, newlines})
		}
		runes++
		if r == '\n' {
			newlines++
		}
	}
	return index
}

// LineForOffset returns the 1-based source line at offset, clamping out-of-range offsets.
func (index *SourceOffsets) LineForOffset(offset int) int {
	offset = max(0, min(offset, len(index.source)))
	checkpoint := index.before(offset)
	return checkpoint.newlines + strings.Count(index.source[checkpoint.byteOffset:offset], "\n") + 1
}

// LineRangeForSpan returns the inclusive line range using the same exclusive-end
// adjustment as the standalone LineRangeForSpan function.
func (index *SourceOffsets) LineRangeForSpan(span ChunkSpan) (int, int) {
	end := span.End
	if end > span.Start {
		end--
	}
	return index.LineForOffset(span.Start), index.LineForOffset(end)
}

// RuneSpan converts valid byte spans to rune offsets, preserving prefix rune
// counts even for an offset within a multibyte sequence or malformed UTF-8.
func (index *SourceOffsets) RuneSpan(startByte, endByte int) (int, int) {
	if startByte < 0 || endByte < 0 || endByte < startByte || endByte > len(index.source) {
		return 0, 0
	}
	return index.runesBefore(startByte), index.runesBefore(endByte)
}

func (index *SourceOffsets) before(offset int) sourceCheckpoint {
	boundary := sort.Search(len(index.checkpoints), func(i int) bool {
		return index.checkpoints[i].byteOffset > offset
	}) - 1
	return index.checkpoints[boundary]
}

func (index *SourceOffsets) runesBefore(offset int) int {
	checkpoint := index.before(offset)
	return checkpoint.runes + utf8.RuneCountInString(index.source[checkpoint.byteOffset:offset])
}

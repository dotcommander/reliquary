package chunking

import (
	"strings"
	"unicode/utf8"
)

// Optimal strategy registers a size-gating pass-through chunker that prefers
// boundary-aware splits only when the accumulated text exceeds OptimalLength/2.
const Optimal Strategy = "optimal"

// Seam markers are in-band protocol strings inserted between non-contiguous
// content samples. Fixed protocol tokens — not user-configurable.
const (
	seamMiddle = "\n\n[... content sample from middle section ...]\n"
	seamEnd    = "\n\n[... content sample from end section ...]\n"
)

// OptimalChunker gates content into size bands and splits large inputs using
// boundary-aware greedy chunking.
//
// Three-tier gate:
//   - len(text) <= MinLength: pass-through (insufficient context for splitting)
//   - len(text) <= MaxLength: pass-through (optimal range for downstream AI)
//   - len(text) > MaxLength:  StrategicSample (head/midpoint/tail with seam markers)
//
// OptimalLength, MinLength, and MaxLength are caller-supplied;
// NewOptimalChunker provides research-backed defaults.
type OptimalChunker struct {
	OptimalLength int // target chunk size for SplitIntoChunks
	MinLength     int // below this, content passes through unchanged
	MaxLength     int // above this, StrategicSample is applied
}

// NewOptimalChunker creates an OptimalChunker with defaults based on AI
// behavior research: OptimalLength 10000, MinLength 5000, MaxLength 15000.
func NewOptimalChunker() *OptimalChunker {
	return &OptimalChunker{
		OptimalLength: 10000,
		MinLength:     5000,
		MaxLength:     15000,
	}
}

// Strategy returns the Optimal strategy constant.
func (c *OptimalChunker) Strategy() Strategy { return Optimal }

// Chunk satisfies the Chunker interface. size maps to OptimalLength; overlap
// is ignored (reserved for future spec). Returns nil for empty text or
// non-positive size.
func (c *OptimalChunker) Chunk(text string, size int, _ int) []Chunk {
	if text == "" || size <= 0 {
		return nil
	}
	c2 := &OptimalChunker{
		OptimalLength: size,
		MinLength:     c.MinLength,
		MaxLength:     c.MaxLength,
	}
	spans := c2.splitIntoSpans(text)
	chunks := make([]Chunk, len(spans))
	for i, sp := range spans {
		chunks[i] = buildChunkWithSpan(i, sp.text, sp.start, sp.end)
	}
	return chunks
}

// ChunkContent applies three-tier gating and returns a single string:
//   - below MinLength: input unchanged
//   - below MaxLength: input unchanged
//   - above MaxLength: StrategicSample output with seam markers
//
// Size comparisons use rune counts to match the Chunker interface semantics.
func (c *OptimalChunker) ChunkContent(text string) string {
	if utf8.RuneCountInString(text) <= c.MinLength {
		return text
	}
	if utf8.RuneCountInString(text) <= c.MaxLength {
		return text
	}
	return StrategicSample(text, c.OptimalLength)
}

// SplitIntoChunks divides content greedily into chunks of at most
// OptimalLength runes. It prefers \n\n (paragraph) or ". " (sentence) break
// points only when the break position is past OptimalLength/2, avoiding tiny
// leading chunks. Otherwise it hard-cuts at OptimalLength runes.
func (c *OptimalChunker) SplitIntoChunks(text string) []string {
	spans := c.splitIntoSpans(text)
	parts := make([]string, len(spans))
	for i, sp := range spans {
		parts[i] = sp.text
	}
	return parts
}

// splitIntoSpans divides content greedily into spans of at most OptimalLength
// runes, tracking byte offsets into the original text. Break points (\n\n, ". ")
// are searched in byte space but the window is measured in rune count.
func (c *OptimalChunker) splitIntoSpans(text string) []textSpan {
	if utf8.RuneCountInString(text) <= c.OptimalLength {
		return []textSpan{{text: text, start: 0, end: len(text)}}
	}

	half := c.OptimalLength / 2
	if half == 0 {
		half = 1
	}

	var spans []textSpan
	byteCursor := 0
	runeCursor := 0
	runes := []rune(text)
	totalRunes := len(runes)

	for runeCursor < totalRunes {
		remainingRunes := totalRunes - runeCursor

		if remainingRunes <= c.OptimalLength {
			// Flush remaining text.
			spans = append(spans, textSpan{
				text:  string(runes[runeCursor:]),
				start: byteCursor,
				end:   len(text),
			})
			break
		}

		// Find the byte offset of the (runeCursor + OptimalLength)th rune.
		windowEndByte := byteCursor + runeCountBytes(text[byteCursor:], c.OptimalLength)

		chunkBytes := c.OptimalLength // rune count target
		chunkByteEnd := windowEndByte

		// Try paragraph break in upper half of the window.
		window := text[byteCursor:windowEndByte]

		if idx := strings.LastIndex(window, "\n\n"); idx >= 0 {
			breakByte := byteCursor + idx
			breakRune := runeCursor + len([]rune(text[byteCursor:breakByte]))
			if breakRune > runeCursor+half {
				// Include the \n\n in both byte end and rune advance.
				chunkBytes = breakRune - runeCursor + 2
				chunkByteEnd = breakByte + 2
			}
		} else if idx := strings.LastIndex(window, ". "); idx >= 0 {
			breakByte := byteCursor + idx
			breakRune := runeCursor + len([]rune(text[byteCursor:breakByte]))
			if breakRune > runeCursor+half {
				// Include the ". " in both byte end and rune advance.
				chunkBytes = breakRune - runeCursor + 2
				chunkByteEnd = breakByte + 2
			}
		}

		spans = append(spans, textSpan{
			text:  text[byteCursor:chunkByteEnd],
			start: byteCursor,
			end:   chunkByteEnd,
		})
		runeCursor += chunkBytes
		byteCursor = chunkByteEnd
	}

	return spans
}

// runeCountBytes returns the byte length of the first n runes in s.
func runeCountBytes(s string, n int) int {
	b := 0
	for i := 0; i < n && b < len(s); i++ {
		_, w := utf8.DecodeRuneInString(s[b:])
		b += w
	}
	return b
}

// StrategicSample returns a bounded representation of text by keeping the
// first 67% and splicing midpoint + tail samples (each remainingQuota/3),
// separated by seam markers. Returns text unchanged when the rune count of text
// is less than or equal to optimalLen.
//
// Rune-boundary safety: the function operates on a slice of runes to ensure
// multi-byte UTF-8 codepoints are never split.
func StrategicSample(text string, optimalLen int) string {
	runes := []rune(text)
	if len(runes) <= optimalLen || optimalLen <= 0 {
		return text
	}

	firstPortion := optimalLen * 2 / 3
	remainingQuota := optimalLen - firstPortion
	beginning := runes[:firstPortion]
	remaining := runes[firstPortion:]

	if len(remaining) <= remainingQuota {
		return string(beginning) + string(remaining)
	}

	sampleSize := remainingQuota / 3
	midPoint := len(remaining) / 2
	endPoint := len(remaining) - sampleSize

	// Guard: cap middle sample if it would exceed the remaining text.
	if midPoint+sampleSize > len(remaining) {
		sampleSize = len(remaining) - midPoint
	}

	// Guard: if tail overlaps with middle, skip tail sample.
	if endPoint < midPoint+sampleSize {
		var b strings.Builder
		b.Grow(len(beginning) + len(seamMiddle) + sampleSize)
		b.WriteString(string(beginning))
		b.WriteString(seamMiddle)
		b.WriteString(string(remaining[midPoint : midPoint+sampleSize]))
		return b.String()
	}

	// Guard: zero sample size (very small remainingQuota).
	if sampleSize == 0 {
		var b strings.Builder
		b.WriteString(string(beginning))
		b.WriteString(seamMiddle)
		if half := len(remaining) / 2; half > 0 {
			b.WriteString(string(remaining[:half]))
		}
		return b.String()
	}

	var b strings.Builder
	b.Grow(optimalLen + len(seamMiddle) + len(seamEnd) + 2*sampleSize)
	b.WriteString(string(beginning))
	b.WriteString(seamMiddle)
	b.WriteString(string(remaining[midPoint : midPoint+sampleSize]))
	b.WriteString(seamEnd)
	b.WriteString(string(remaining[endPoint:]))
	return b.String()
}

// StrategicSample on OptimalChunker delegates to the package function.
func (c *OptimalChunker) StrategicSample(text string) string {
	return StrategicSample(text, c.OptimalLength)
}

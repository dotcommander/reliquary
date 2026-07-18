package textutil

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Order controls how FragmentRange trades off exact and whitespace-normalized
// matches once the exact match at the cursor has missed.
type Order int

const (
	// ExactFirst tries an exact match anywhere (from the cursor, then from the
	// start of content) before any normalized match. This is the order the
	// chunking pipeline relies on.
	ExactFirst Order = iota
	// NormalizedEarly tries a normalized match at or after the cursor before
	// falling back to an exact match from the start of content.
	NormalizedEarly
)

// FragmentRange locates fragment in content and returns byte offsets. It first tries an
// exact match at the cursor, then falls back per ord: ExactFirst prefers any
// exact match over a normalized one; NormalizedEarly prefers a normalized match
// at/after the cursor over an exact match earlier in the content. Used by
// chunkers that join lines or collapse paragraph boundaries.
func FragmentRange(content, fragment string, cursor int, ord Order) (int, int, bool) {
	if fragment == "" {
		return 0, 0, false
	}
	if cursor < 0 || cursor > len(content) {
		cursor = 0
	}
	if idx := strings.Index(content[cursor:], fragment); idx >= 0 {
		start := cursor + idx
		return start, start + len(fragment), true
	}
	exactFromStart := func() (int, int, bool) {
		idx := strings.Index(content, fragment)
		if idx < 0 {
			return 0, 0, false
		}
		return idx, idx + len(fragment), true
	}
	normAtCursor := func() (int, int, bool) { return normalizedRange(content, fragment, cursor, false) }
	normFromStart := func() (int, int, bool) { return normalizedRange(content, fragment, 0, true) }
	switch ord {
	case NormalizedEarly:
		if s, e, ok := normAtCursor(); ok {
			return s, e, true
		}
		if s, e, ok := exactFromStart(); ok {
			return s, e, true
		}
		return normFromStart()
	default: // ExactFirst
		if s, e, ok := exactFromStart(); ok {
			return s, e, true
		}
		if s, e, ok := normAtCursor(); ok {
			return s, e, true
		}
		return normFromStart()
	}
}

func normalizedRange(content, fragment string, cursor int, allowBeforeCursor bool) (int, int, bool) {
	contentNorm := normalizeWithMap(content)
	fragmentNorm := normalize(fragment)
	if fragmentNorm == "" {
		return 0, 0, false
	}

	normCursor := normalizedCursor(contentNorm, cursor)
	if idx := strings.Index(contentNorm.text[normCursor:], fragmentNorm); idx >= 0 {
		return contentNorm.byteRange(normCursor+idx, len(fragmentNorm))
	}
	if allowBeforeCursor {
		idx := strings.Index(contentNorm.text, fragmentNorm)
		if idx < 0 {
			return 0, 0, false
		}
		return contentNorm.byteRange(idx, len(fragmentNorm))
	}
	return 0, 0, false
}

func normalizedCursor(n normalizedText, cursor int) int {
	for i, start := range n.starts {
		if start >= cursor {
			return i
		}
	}
	return len(n.text)
}

type normalizedText struct {
	text   string
	starts []int
	ends   []int
}

func (n normalizedText) byteRange(normStart, normLen int) (int, int, bool) {
	if normStart < 0 || normLen <= 0 || normStart+normLen > len(n.starts) {
		return 0, 0, false
	}
	return n.starts[normStart], n.ends[normStart+normLen-1], true
}

func normalizeWithMap(s string) normalizedText {
	var b strings.Builder
	starts := make([]int, 0, len(s))
	ends := make([]int, 0, len(s))
	pendingSpace := false

	for bytePos, r := range s {
		width := utf8.RuneLen(r)
		if width < 0 {
			width = 1
		}
		if unicode.IsSpace(r) {
			pendingSpace = b.Len() > 0
			continue
		}
		if pendingSpace {
			b.WriteByte(' ')
			starts = append(starts, bytePos)
			ends = append(ends, bytePos)
			pendingSpace = false
		}
		normStart := b.Len()
		b.WriteRune(r)
		for i := normStart; i < b.Len(); i++ {
			starts = append(starts, bytePos)
			ends = append(ends, bytePos+width)
		}
	}
	return normalizedText{text: b.String(), starts: starts, ends: ends}
}

func normalize(s string) string {
	return normalizeWithMap(s).text
}

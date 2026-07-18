// Package document defines provider-neutral document primitives.
package document

import (
	"strings"
	"unicode/utf8"
)

// Metadata carries caller-owned document attributes.
type Metadata map[string]string

// Format identifies a document or span format without owning parsing policy.
type Format string

const (
	FormatText     Format = "text"
	FormatMarkdown Format = "markdown"
	FormatHTML     Format = "html"
	FormatPDF      Format = "pdf"
)

// Document is normalized text plus structural metadata.
type Document struct {
	ID       string
	Title    string
	Format   Format
	Text     string
	Sections []Section
	Metadata Metadata
}

// Section identifies a named range of document text.
type Section struct {
	ID    string
	Title string
	Span  Span
}

// ElementKind identifies generic structure detected before chunking.
type ElementKind string

const (
	ElementKindTitle     ElementKind = "title"
	ElementKindNarrative ElementKind = "narrative_text"
	ElementKindListItem  ElementKind = "list_item"
	ElementKindTable     ElementKind = "table"
	ElementKindCode      ElementKind = "code"
	ElementKindPageBreak ElementKind = "page_break"
	ElementKindUnknown   ElementKind = "unknown"
)

// ParserStrategy labels the caller-owned parser strategy that produced an
// element. It is descriptive only and does not select parser behavior.
type ParserStrategy string

const (
	ParserStrategyPlainText ParserStrategy = "plain_text"
	ParserStrategyMarkdown  ParserStrategy = "markdown"
	ParserStrategyHTML      ParserStrategy = "html"
	ParserStrategyPDF       ParserStrategy = "pdf"
	ParserStrategyUnknown   ParserStrategy = "unknown"
)

// Element is a span-backed, provider-neutral unit detected before chunking.
type Element struct {
	ID       string
	Kind     ElementKind
	Strategy ParserStrategy
	Span     Span
	Text     string
	Metadata Metadata
}

// Span is a byte-offset range into a UTF-8 string.
type Span struct {
	StartByte int
	EndByte   int
}

// Valid returns true when the span is inside text and aligned to rune boundaries.
func (s Span) Valid(text string) bool {
	if s.StartByte < 0 || s.EndByte < s.StartByte || s.EndByte > len(text) {
		return false
	}
	startOK := s.StartByte == len(text) || utf8.RuneStart(text[s.StartByte])
	endOK := s.EndByte == len(text) || utf8.RuneStart(text[s.EndByte])
	return startOK && endOK
}

// Text returns the substring covered by the span.
func (s Span) Text(text string) (string, bool) {
	if !s.Valid(text) {
		return "", false
	}
	return text[s.StartByte:s.EndByte], true
}

// ElementText returns the element text, preferring the span when it is valid
// for source and falling back to Element.Text.
func (e Element) ElementText(source string) string {
	if text, ok := e.Span.Text(source); ok {
		return text
	}
	return e.Text
}

// NormalizeText normalizes line endings and strips a UTF-8 BOM.
func NormalizeText(text string) string {
	text = strings.TrimPrefix(text, "\ufeff")
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.ReplaceAll(text, "\r", "\n")
}

// ByteOffsetForRune returns the byte offset for a rune index.
func ByteOffsetForRune(text string, runeIndex int) (int, bool) {
	if runeIndex < 0 {
		return 0, false
	}
	if runeIndex == 0 {
		return 0, true
	}
	i := 0
	for offset := range text {
		if i == runeIndex {
			return offset, true
		}
		i++
	}
	if i == runeIndex {
		return len(text), true
	}
	return 0, false
}

// RuneOffsetForByte returns the rune index for a byte offset.
func RuneOffsetForByte(text string, byteOffset int) (int, bool) {
	if byteOffset < 0 || byteOffset > len(text) {
		return 0, false
	}
	if byteOffset == len(text) {
		return utf8.RuneCountInString(text), true
	}
	if !utf8.RuneStart(text[byteOffset]) {
		return 0, false
	}
	return utf8.RuneCountInString(text[:byteOffset]), true
}

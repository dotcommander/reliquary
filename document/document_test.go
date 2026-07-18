package document

import "testing"

func TestNormalizeText(t *testing.T) {
	t.Parallel()

	got := NormalizeText("\ufeffa\r\nb\rc")
	if got != "a\nb\nc" {
		t.Fatalf("got %q", got)
	}
}

func TestOffsetsAndSpan(t *testing.T) {
	t.Parallel()

	text := "aéz"
	offset, ok := ByteOffsetForRune(text, 2)
	if !ok || offset != 3 {
		t.Fatalf("byte offset = %d %v, want 3 true", offset, ok)
	}
	runes, ok := RuneOffsetForByte(text, 3)
	if !ok || runes != 2 {
		t.Fatalf("rune offset = %d %v, want 2 true", runes, ok)
	}
	got, ok := (Span{StartByte: 1, EndByte: 3}).Text(text)
	if !ok || got != "é" {
		t.Fatalf("span text = %q %v, want é true", got, ok)
	}
	if !(Span{StartByte: len(text), EndByte: len(text)}).Valid(text) {
		t.Fatal("empty end span should be valid")
	}
}

func TestElementTextPrefersValidSpan(t *testing.T) {
	t.Parallel()

	source := "Title\n\nbody"
	element := Element{
		ID:       "e1",
		Kind:     ElementKindTitle,
		Strategy: ParserStrategyMarkdown,
		Span:     Span{StartByte: 0, EndByte: 5},
		Text:     "fallback",
		Metadata: Metadata{"level": "1"},
	}
	if got := element.ElementText(source); got != "Title" {
		t.Fatalf("ElementText = %q, want Title", got)
	}

	element.Span = Span{StartByte: 99, EndByte: 100}
	if got := element.ElementText(source); got != "fallback" {
		t.Fatalf("fallback ElementText = %q, want fallback", got)
	}
}

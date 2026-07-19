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

func TestByteOffsetForRuneEdgeCases(t *testing.T) {
	t.Parallel()

	text := "aéz" // len 4 bytes, 3 runes: 'a' (1 byte), 'é' (2 bytes: 0xc3 0xa9), 'z' (1 byte)

	// Negative rune index
	if _, ok := ByteOffsetForRune(text, -1); ok {
		t.Fatal("expected false for negative rune index")
	}

	// Zero index
	if off, ok := ByteOffsetForRune(text, 0); !ok || off != 0 {
		t.Fatalf("index 0: off=%d ok=%v, want 0 true", off, ok)
	}

	// Index equal to rune count
	if off, ok := ByteOffsetForRune(text, 3); !ok || off != len(text) {
		t.Fatalf("index 3: off=%d ok=%v, want %d true", off, ok, len(text))
	}

	// Out of bounds rune index
	if _, ok := ByteOffsetForRune(text, 4); ok {
		t.Fatal("expected false for out of bounds rune index")
	}
}

func TestRuneOffsetForByteEdgeCases(t *testing.T) {
	t.Parallel()

	text := "aéz" // byte 0:'a', byte 1:0xc3, byte 2:0xa9, byte 3:'z', byte 4:EOF

	// Negative byte offset
	if _, ok := RuneOffsetForByte(text, -1); ok {
		t.Fatal("expected false for negative byte offset")
	}

	// Out of bounds byte offset
	if _, ok := RuneOffsetForByte(text, 10); ok {
		t.Fatal("expected false for out of bounds byte offset")
	}

	// Byte offset equal to len(text)
	if r, ok := RuneOffsetForByte(text, len(text)); !ok || r != 3 {
		t.Fatalf("offset len: r=%d ok=%v, want 3 true", r, ok)
	}

	// Byte offset pointing inside a multi-byte rune (continuation byte)
	if _, ok := RuneOffsetForByte(text, 2); ok {
		t.Fatal("expected false for non-rune start byte offset")
	}
}

func TestSpanValidEdgeCases(t *testing.T) {
	t.Parallel()

	text := "aéz"

	// Start < 0
	if (Span{StartByte: -1, EndByte: 2}).Valid(text) {
		t.Fatal("expected invalid for negative StartByte")
	}

	// End < Start
	if (Span{StartByte: 2, EndByte: 1}).Valid(text) {
		t.Fatal("expected invalid for EndByte < StartByte")
	}

	// End > len
	if (Span{StartByte: 0, EndByte: 10}).Valid(text) {
		t.Fatal("expected invalid for EndByte > len")
	}

	// Unaligned start/end
	if (Span{StartByte: 2, EndByte: 3}).Valid(text) {
		t.Fatal("expected invalid for unaligned start byte")
	}

	// Invalid span text call returns ("", false)
	if _, ok := (Span{StartByte: 2, EndByte: 3}).Text(text); ok {
		t.Fatal("Text() on invalid span returned ok=true")
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

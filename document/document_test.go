package document

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func TestFromReader(t *testing.T) {
	t.Parallel()

	metadata := map[string]string{
		"source":      "test",
		"filename":    "metadata-name.txt",
		"document_id": "metadata-id",
	}
	option := WithMetadata(metadata)
	metadata["source"] = "changed"

	doc, err := FromReader(
		"doc-1",
		strings.NewReader("\ufefffirst\r\nsecond\rthird"),
		WithFilename("notes.md"),
		WithFormat(FormatMarkdown),
		option,
	)
	if err != nil {
		t.Fatalf("FromReader() error = %v", err)
	}
	if doc.ID != "doc-1" {
		t.Fatalf("ID = %q, want doc-1", doc.ID)
	}
	if doc.Title != "notes.md" {
		t.Fatalf("Title = %q, want notes.md", doc.Title)
	}
	if doc.Format != FormatMarkdown {
		t.Fatalf("Format = %q, want markdown", doc.Format)
	}
	if doc.Text != "first\nsecond\nthird" {
		t.Fatalf("Text = %q, want normalized text", doc.Text)
	}
	if got := doc.Metadata["source"]; got != "test" {
		t.Fatalf("Metadata[source] = %q, want snapshotted value", got)
	}
	if got := doc.Metadata["filename"]; got != "metadata-name.txt" {
		t.Fatalf("Metadata[filename] = %q, want caller value preserved", got)
	}
	if got := doc.Metadata["document_id"]; got != "metadata-id" {
		t.Fatalf("Metadata[document_id] = %q, want caller value preserved", got)
	}

	second, err := FromReader("doc-2", strings.NewReader("text"), option)
	if err != nil {
		t.Fatalf("second FromReader() error = %v", err)
	}
	doc.Metadata["source"] = "first-document-change"
	if got := second.Metadata["source"]; got != "test" {
		t.Fatalf("reused option shared metadata: got %q", got)
	}
}

func TestFromReaderDefaults(t *testing.T) {
	t.Parallel()

	doc, err := FromReader(
		"doc-1",
		strings.NewReader("plain text"),
		WithFilename("page.md"),
		WithMetadata(map[string]string{"source": "test"}),
	)
	if err != nil {
		t.Fatalf("FromReader() error = %v", err)
	}
	if doc.Format != FormatText {
		t.Fatalf("Format = %q, want text", doc.Format)
	}
	if _, ok := doc.Metadata["filename"]; ok {
		t.Fatal("metadata contains injected filename")
	}
	if _, ok := doc.Metadata["document_id"]; ok {
		t.Fatal("metadata contains injected document_id")
	}
}

func TestFromReaderLimits(t *testing.T) {
	t.Parallel()

	t.Run("exact limit", func(t *testing.T) {
		doc, err := FromReader("exact", strings.NewReader("é"), WithMaxBytes(2))
		if err != nil {
			t.Fatalf("FromReader() error = %v", err)
		}
		if doc.Text != "é" {
			t.Fatalf("Text = %q, want é", doc.Text)
		}
	})

	t.Run("oversized", func(t *testing.T) {
		_, err := FromReader("large", strings.NewReader("abc"), WithMaxBytes(2))
		if !errors.Is(err, ErrInputTooLarge) {
			t.Fatalf("error = %v, want ErrInputTooLarge", err)
		}
	})

	t.Run("default oversized", func(t *testing.T) {
		input := strings.Repeat("x", int(defaultMaxReaderBytes)+1)
		_, err := FromReader("large", strings.NewReader(input))
		if !errors.Is(err, ErrInputTooLarge) {
			t.Fatalf("error = %v, want ErrInputTooLarge", err)
		}
	})

	for _, limit := range []int64{0, -1} {
		limit := limit
		t.Run("invalid limit", func(t *testing.T) {
			_, err := FromReader("invalid", strings.NewReader("text"), WithMaxBytes(limit))
			if err == nil {
				t.Fatalf("WithMaxBytes(%d) error = nil", limit)
			}
		})
	}
}

func TestFromReaderRejectsInvalidInput(t *testing.T) {
	t.Parallel()

	t.Run("nil reader", func(t *testing.T) {
		_, err := FromReader("nil", nil)
		if err == nil {
			t.Fatal("FromReader() error = nil")
		}
	})

	t.Run("invalid UTF-8", func(t *testing.T) {
		_, err := FromReader("invalid", strings.NewReader(string([]byte{0xff})))
		if !errors.Is(err, ErrInvalidUTF8) {
			t.Fatalf("error = %v, want ErrInvalidUTF8", err)
		}
	})

	t.Run("reader error", func(t *testing.T) {
		readerErr := errors.New("reader failed")
		_, err := FromReader("failure", errorReader{err: readerErr})
		if !errors.Is(err, readerErr) {
			t.Fatalf("error = %v, want wrapped reader error", err)
		}
	})
}

type errorReader struct {
	err error
}

func (r errorReader) Read([]byte) (int, error) {
	return 0, r.err
}

var _ io.Reader = errorReader{}

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

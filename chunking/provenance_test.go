package chunking

import (
	"testing"
)

func TestResolveChunkSpan_ValidSpanPassthrough(t *testing.T) {
	t.Parallel()
	content := "Hello world this is a test"
	chunk := Chunk{
		Text:      "this is a test",
		StartChar: 12,
		EndChar:   26,
	}
	span, ok := ResolveChunkSpan(content, chunk, 0)
	if !ok {
		t.Fatal("expected ok")
	}
	if span.Start != 12 || span.End != 26 {
		t.Fatalf("span = [%d, %d), want [12, 26)", span.Start, span.End)
	}
	if content[span.Start:span.End] != chunk.Text {
		t.Fatalf("content[Start:End] = %q, want %q", content[span.Start:span.End], chunk.Text)
	}
}

func TestResolveChunkSpan_InvalidSpanFallsBackToLocate(t *testing.T) {
	t.Parallel()
	content := "Alpha beta gamma delta"
	chunk := Chunk{
		Text:      "gamma",
		StartChar: 99, // wrong
		EndChar:   99, // wrong
	}
	span, ok := ResolveChunkSpan(content, chunk, 0)
	if !ok {
		t.Fatal("expected ok")
	}
	if content[span.Start:span.End] != "gamma" {
		t.Fatalf("content[Start:End] = %q, want %q", content[span.Start:span.End], "gamma")
	}
}

func TestResolveChunkSpan_ZeroStartValidWhenEndNonZero(t *testing.T) {
	t.Parallel()
	content := "Begin middle end"
	chunk := Chunk{
		Text:      "Begin",
		StartChar: 0,
		EndChar:   5,
	}
	span, ok := ResolveChunkSpan(content, chunk, 0)
	if !ok {
		t.Fatal("expected ok — zero start is valid when end is non-zero")
	}
	if span.Start != 0 || span.End != 5 {
		t.Fatalf("span = [%d, %d), want [0, 5)", span.Start, span.End)
	}
}

func TestResolveChunkSpan_WhitespaceNormalizedFallback(t *testing.T) {
	t.Parallel()
	// Source has multiple spaces/newlines; fragment has collapsed whitespace.
	content := "Hello   world\n\nfoo bar"
	chunk := Chunk{
		Text: "Hello world", // collapsed whitespace
		// Zero span → triggers Locate → normalized fallback
	}
	span, ok := ResolveChunkSpan(content, chunk, 0)
	if !ok {
		t.Fatal("expected ok via whitespace-normalized fallback")
	}
	// The span maps to the original source region. Locate maps back to source
	// bytes, so the source slice will have the original whitespace.
	// We verify the span is valid and non-zero.
	if span.Start == 0 && span.End == 0 {
		t.Fatal("expected non-zero span")
	}
	if span.End <= span.Start {
		t.Fatalf("invalid span [%d, %d)", span.Start, span.End)
	}
}

func TestResolveChunkSpan_RepeatedFragmentsUseCursor(t *testing.T) {
	t.Parallel()
	content := "abc abc abc"

	chunk := Chunk{Text: "abc"}
	span1, ok := ResolveChunkSpan(content, chunk, 0)
	if !ok {
		t.Fatal("first resolve: expected ok")
	}
	if span1.Start != 0 {
		t.Fatalf("first span start = %d, want 0", span1.Start)
	}

	cursor := NextChunkCursor(span1)
	span2, ok := ResolveChunkSpan(content, chunk, cursor)
	if !ok {
		t.Fatal("second resolve: expected ok")
	}
	if span2.Start == span1.Start {
		t.Fatalf("second span start = %d, same as first — cursor should advance past start", span2.Start)
	}
	if span2.Start != 4 {
		t.Fatalf("second span start = %d, want 4", span2.Start)
	}

	cursor = NextChunkCursor(span2)
	span3, ok := ResolveChunkSpan(content, chunk, cursor)
	if !ok {
		t.Fatal("third resolve: expected ok")
	}
	if span3.Start != 8 {
		t.Fatalf("third span start = %d, want 8", span3.Start)
	}
}

func TestResolveChunkSpan_UnicodeByteOffsets(t *testing.T) {
	t.Parallel()
	content := "Hello 世界 goodbye"
	// "世界" is 6 bytes in UTF-8 (3 bytes each).
	chunk := Chunk{
		Text: "世界",
		// No span → will Locate
	}
	span, ok := ResolveChunkSpan(content, chunk, 0)
	if !ok {
		t.Fatal("expected ok")
	}
	if content[span.Start:span.End] != "世界" {
		t.Fatalf("content[Start:End] = %q, want %q", content[span.Start:span.End], "世界")
	}
	// Byte offset should be 6 (after "Hello " = 6 bytes).
	if span.Start != 6 {
		t.Fatalf("span.Start = %d, want 6", span.Start)
	}
	if span.End != 12 {
		t.Fatalf("span.End = %d, want 12", span.End)
	}
}

func TestLineRangeForSpan_EndAtNewlineUsesPreviousLine(t *testing.T) {
	t.Parallel()
	content := "line one\nline two\nline three\n"
	// Span covering "line one" — bytes [0, 8)
	span := ChunkSpan{Start: 0, End: 8}
	startLine, endLine := LineRangeForSpan(content, span)
	if startLine != 1 || endLine != 1 {
		t.Fatalf("lines = [%d, %d], want [1, 1]", startLine, endLine)
	}

	// Span covering "line one\n" — bytes [0, 9), end at newline.
	// endOffset = 9 - 1 = 8 → byte 8 is '\n', LineForOffset counts \n
	// at offset 8: content[:8] = "line one" → 0 newlines → line 1.
	span2 := ChunkSpan{Start: 0, End: 9}
	startLine2, endLine2 := LineRangeForSpan(content, span2)
	if startLine2 != 1 || endLine2 != 1 {
		t.Fatalf("lines = [%d, %d], want [1, 1] (newline should not push to line 2)", startLine2, endLine2)
	}
}

func TestLineRangeForSpan_MultilineChunk(t *testing.T) {
	t.Parallel()
	content := "first line\nsecond line\nthird line"
	// Span covering "first line\nsecond line" — bytes [0, 22)
	span := ChunkSpan{Start: 0, End: 22}
	startLine, endLine := LineRangeForSpan(content, span)
	if startLine != 1 {
		t.Fatalf("startLine = %d, want 1", startLine)
	}
	// endOffset = 22 - 1 = 21 → content[:21] has one newline → line 2.
	if endLine != 2 {
		t.Fatalf("endLine = %d, want 2", endLine)
	}
}

func TestResolveChunkSpan_EmptyChunkFalse(t *testing.T) {
	t.Parallel()
	content := "some content"
	chunk := Chunk{Text: ""}
	_, ok := ResolveChunkSpan(content, chunk, 0)
	if ok {
		t.Fatal("empty chunk text should not resolve")
	}
}

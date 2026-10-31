package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSourceOffsetsMatchesStandaloneProjections(t *testing.T) {
	t.Parallel()
	for _, source := range []string{"", "a\nb\n", "é🚀\n\xff\xc3x\nend"} {
		index := NewSourceOffsets(source)
		for start := -1; start <= len(source)+1; start++ {
			if got, want := index.LineForOffset(start), LineForOffset(source, start); got != want {
				t.Fatalf("source %q line at %d = %d, want %d", source, start, got, want)
			}
			for end := -1; end <= len(source)+1; end++ {
				span := ChunkSpan{Start: start, End: end}
				gotStart, gotEnd := index.LineRangeForSpan(span)
				wantStart, wantEnd := LineRangeForSpan(source, span)
				if gotStart != wantStart || gotEnd != wantEnd {
					t.Fatalf("source %q lines for %+v = %d-%d, want %d-%d", source, span, gotStart, gotEnd, wantStart, wantEnd)
				}
				gotStart, gotEnd = index.RuneSpan(start, end)
				wantStart, wantEnd = RuneSpan(source, start, end)
				if gotStart != wantStart || gotEnd != wantEnd {
					t.Fatalf("source %q runes for %+v = %d-%d, want %d-%d", source, span, gotStart, gotEnd, wantStart, wantEnd)
				}
			}
		}
	}
}

func TestSourceOffsetsSparseCheckpoints(t *testing.T) {
	t.Parallel()
	for _, source := range []string{
		strings.Repeat("a", 2048),
		strings.Repeat("\n", 2048),
		strings.Repeat("é🚀\n\xff\xc3x", 200),
	} {
		index := NewSourceOffsets(source)
		if got, maxEntries := len(index.checkpoints), 1+len(source)/sourceCheckpointSpacing; got > maxEntries {
			t.Fatalf("%d checkpoints exceed bound %d", got, maxEntries)
		}
		for offset := 0; offset <= len(source); offset++ {
			if got, want := index.LineForOffset(offset), LineForOffset(source, offset); got != want {
				t.Fatalf("line at %d = %d, want %d", offset, got, want)
			}
			got, _ := index.RuneSpan(offset, len(source))
			if want := utf8.RuneCountInString(source[:offset]); got != want {
				t.Fatalf("runes at %d = %d, want %d", offset, got, want)
			}
		}
	}
}

func TestSourceOffsetsInteriorUTF8Prefix(t *testing.T) {
	t.Parallel()
	source := "🚀\xff"
	index := NewSourceOffsets(source)
	for offset := 0; offset <= len(source); offset++ {
		start, end := index.RuneSpan(offset, len(source))
		if want := utf8.RuneCountInString(source[:offset]); start != want || end != 2 {
			t.Fatalf("rune span at %d = %d-%d, want %d-2", offset, start, end, want)
		}
	}
}

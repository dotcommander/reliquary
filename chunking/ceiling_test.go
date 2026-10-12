package chunking

import (
	"strings"
	"testing"
)

func TestEnforceTokenCeilingSplitsOversizedChunks(t *testing.T) {
	t.Parallel()
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 60)
	base, err := NewChunker(SmartBoundary)
	if err != nil {
		t.Fatalf("NewChunker: %v", err)
	}
	chunks := base.Chunk(text, 10000, 0)
	if len(chunks) == 0 {
		t.Fatal("expected at least one base chunk")
	}
	limited, err := EnforceTokenCeiling(chunks, "cl100k_base", 16)
	if err != nil {
		t.Fatalf("EnforceTokenCeiling: %v", err)
	}
	if len(limited) <= len(chunks) {
		t.Fatalf("expected ceiling to split chunks: got %d from %d", len(limited), len(chunks))
	}
	for i, c := range limited {
		if c.TokenCount <= 0 || c.TokenCount > 16 {
			t.Fatalf("chunk %d token count %d outside (0, 16]", i, c.TokenCount)
		}
	}
}

func TestEnforceTokenCeilingFillsCountsWithoutCeiling(t *testing.T) {
	t.Parallel()
	text := "One sentence. Another sentence. A third sentence for good measure."
	base, err := NewChunker(SentenceBoundary)
	if err != nil {
		t.Fatalf("NewChunker: %v", err)
	}
	chunks := base.Chunk(text, 2000, 0)
	limited, err := EnforceTokenCeiling(chunks, "", 0)
	if err != nil {
		t.Fatalf("EnforceTokenCeiling with zero ceiling: %v", err)
	}
	if len(limited) != len(chunks) {
		t.Fatalf("zero ceiling must not split: got %d from %d", len(limited), len(chunks))
	}
	for i, c := range limited {
		if c.TokenCount <= 0 {
			t.Fatalf("chunk %d missing TokenCount", i)
		}
	}
}

func TestEnforceTokenCeilingRejectsUnknownEncoding(t *testing.T) {
	t.Parallel()
	if _, err := EnforceTokenCeiling(nil, "not_an_encoding", 8); err == nil {
		t.Fatal("expected error for unknown encoding")
	}
}

func TestRuneSpanConvertsByteOffsetsToRunes(t *testing.T) {
	t.Parallel()
	source := "héllo wörld"
	// "héllo" occupies bytes 0..6 (é is 2 bytes) and runes 0..5.
	start, end := RuneSpan(source, 0, 6)
	if start != 0 || end != 5 {
		t.Fatalf("RuneSpan(0,6) = (%d, %d), want (0, 5)", start, end)
	}
	start, end = RuneSpan(source, 6, 11)
	if start != 5 || end != 9 {
		t.Fatalf("RuneSpan(6,11) = (%d, %d), want (5, 9)", start, end)
	}
}

func TestRuneSpanNormalizesInvalidSpans(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name               string
		start, end         int
		wantStart, wantEnd int
	}{
		{"negative start", -1, 4, 0, 0},
		{"end before start", 4, 2, 0, 0},
		{"end past source", 2, 999, 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			start, end := RuneSpan("sample text", tc.start, tc.end)
			if start != tc.wantStart || end != tc.wantEnd {
				t.Fatalf("RuneSpan(%d, %d) = (%d, %d), want (%d, %d)",
					tc.start, tc.end, start, end, tc.wantStart, tc.wantEnd)
			}
		})
	}
	if start, end := RuneSpan("", 0, 0); start != 0 || end != 0 {
		t.Fatalf("RuneSpan on empty source = (%d, %d), want (0, 0)", start, end)
	}
}

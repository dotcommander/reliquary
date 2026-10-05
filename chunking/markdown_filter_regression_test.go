package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestMarkdownAwareFilterPreservesSurroundingProse(t *testing.T) {
	t.Parallel()
	before := "Before prose has enough useful words to survive filtering."
	between := "Between prose also has enough useful words to survive filtering."
	after := "After prose has enough useful words to survive filtering."
	text := before + "\n\n```go\nCODE_SENTINEL := 1\n```\n\n" + between + "\n\n| name | value |\n| --- | --- |\n| TABLE_SENTINEL | 42 |\n\n" + after
	for _, size := range []int{4096, 48} {
		t.Run(stringSizeLabel(size), func(t *testing.T) {
			t.Parallel()
			chunks := newMarkdownAwareChunker().Chunk(text, size, 0)
			seenCode, seenTable := false, false
			for _, chunk := range chunks {
				kind := chunk.Metadata["type"]
				switch kind {
				case "code":
					seenCode = true
				case "table":
					seenTable = true
				case "paragraph":
				default:
					t.Errorf("missing classification for %q", chunk.Text)
				}
				if strings.Contains(chunk.Text, "CODE_SENTINEL") && kind != "code" {
					t.Errorf("code chunk classified as %q", kind)
				}
				if strings.Contains(chunk.Text, "TABLE_SENTINEL") && kind != "table" {
					t.Errorf("table chunk classified as %q", kind)
				}
				if (kind == "code" || kind == "table") && strings.Contains(chunk.Text, "prose") {
					t.Errorf("prose mixed into %s chunk: %q", kind, chunk.Text)
				}
			}
			if !seenCode || !seenTable {
				t.Fatalf("block content missing: code=%v table=%v", seenCode, seenTable)
			}
			var prose []string
			for _, chunk := range FilterProse(chunks) {
				if chunk.Metadata["type"] != "paragraph" {
					t.Errorf("non-prose chunk survived filter: %q", chunk.Text)
				}
				prose = append(prose, chunk.Text)
			}
			got := strings.Join(strings.Fields(strings.Join(prose, " ")), " ")
			want := before + " " + between + " " + after
			if got != want {
				t.Errorf("filtered prose = %q, want %q", got, want)
			}
		})
	}
}

func stringSizeLabel(size int) string {
	if size == 4096 {
		return "packed"
	}
	return "split"
}

func TestMarkdownAwareOversizedBlocksRetainClassification(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		kind string
		text string
	}{
		{"code", "```go\n" + strings.Repeat("界", 180) + "\n```"},
		{"table", "| key | value |\n| --- | --- |\n| x | " + strings.Repeat("界", 180) + " |"},
	} {
		t.Run(tc.kind, func(t *testing.T) {
			t.Parallel()
			chunks := newMarkdownAwareChunker().Chunk(tc.text, 40, 0)
			if len(chunks) < 2 {
				t.Fatalf("want split oversized %s", tc.kind)
			}
			var payload int
			for _, chunk := range chunks {
				if chunk.Metadata["type"] != tc.kind {
					t.Errorf("fragment lost %s classification: %q", tc.kind, chunk.Text)
				}
				if utf8.RuneCountInString(chunk.Text) > 40 {
					t.Errorf("oversized fragment: %q", chunk.Text)
				}
				payload += strings.Count(chunk.Text, "界")
			}
			if payload != 180 {
				t.Errorf("payload runes = %d, want 180", payload)
			}
			if len(FilterProse(chunks)) != 0 {
				t.Error("classified block fragments survived prose filtering")
			}
		})
	}
}

func TestHardLimitPreservesBlockMetadataOnFragments(t *testing.T) {
	t.Parallel()
	for _, kind := range []string{"code", "table"} {
		t.Run(kind, func(t *testing.T) {
			t.Parallel()
			text := strings.Repeat("界", 125)
			chunk := buildChunkWithSpan(0, text, 0, len(text))
			chunk.Metadata = map[string]string{"type": kind, "language": "go"}
			for _, source := range []string{"", text} {
				chunks := EnforceHardLimits([]Chunk{chunk}, LimitOptions{MaxChars: 40, OriginalText: source})
				if len(chunks) != 4 {
					t.Fatalf("got %d fragments, want 4", len(chunks))
				}
				for _, part := range chunks {
					if part.Metadata["type"] != kind || part.Metadata["language"] != "go" {
						t.Errorf("lost metadata: %#v", part.Metadata)
					}
				}
				chunks[0].Metadata["type"] = "changed"
				if chunk.Metadata["type"] != kind || chunks[1].Metadata["type"] != kind {
					t.Error("fragment metadata aliases the input or a sibling")
				}
			}
		})
	}
}

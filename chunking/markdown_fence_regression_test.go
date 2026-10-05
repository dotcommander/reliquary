package chunking

import (
	"strings"
	"testing"
)

func TestMarkdownFenceSpansMatchAdjacentDelimiters(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		text string
		want []string
	}{
		{"adjacent", "```go\nfirst := 1\n```\n\n~~~text\nsecond value\n~~~\n\n```\nthird value\n```\n", []string{"```go\nfirst := 1\n```", "~~~text\nsecond value\n~~~", "```\nthird value\n```"}},
		{"long_fences", "~~~~go\n``` is code\n~~~~~\n\n````go\n~~~ is code\n`````\n", []string{"~~~~go\n``` is code\n~~~~~", "````go\n~~~ is code\n`````"}},
		{"crlf", "~~~go\r\nvalue := 1\r\n~~~\r\n", []string{"~~~go\r\nvalue := 1\r\n~~~"}},
		{"unclosed", "~~~go\nvalue := 1", []string{"~~~go\nvalue := 1"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			blocks := extractMarkdownBlocks([]byte(tc.text))
			if len(blocks) != len(tc.want) {
				t.Fatalf("got %d blocks, want %d", len(blocks), len(tc.want))
			}
			for i, block := range blocks {
				if block.blockType != "code" || block.text != tc.want[i] {
					t.Errorf("block %d = %q (%s), want %q", i, block.text, block.blockType, tc.want[i])
				}
				if block.startByte < 0 || block.endByte <= block.startByte || block.endByte > len(tc.text) {
					t.Fatalf("invalid block span %d:%d", block.startByte, block.endByte)
				}
				if tc.text[block.startByte:block.endByte] != block.text {
					t.Errorf("block %d source slice differs", i)
				}
			}
			chunks := newMarkdownAwareChunker().Chunk(tc.text, 4096, 0)
			if len(chunks) != len(tc.want) {
				t.Fatalf("got %d chunks, want %d", len(chunks), len(tc.want))
			}
			for i, chunk := range chunks {
				if chunk.Text != tc.want[i] || chunk.EndChar <= chunk.StartChar || tc.text[chunk.StartChar:chunk.EndChar] != chunk.Text {
					t.Errorf("chunk %d lost exact fenced source span: %#v", i, chunk)
				}
			}
		})
	}
}

func TestMarkdownSplitFenceSpansRemainTruthful(t *testing.T) {
	t.Parallel()
	for _, fence := range []string{"```", "~~~"} {
		t.Run(fence, func(t *testing.T) {
			t.Parallel()
			text := fence + "go\n" + strings.Repeat("alpha beta gamma delta\n", 8) + fence + "\n"
			chunks := newMarkdownAwareChunker().Chunk(text, 35, 0)
			if len(chunks) < 2 {
				t.Fatal("expected split code fence")
			}
			for _, chunk := range chunks {
				if chunk.Metadata["type"] != "code" {
					t.Errorf("split fence lost code type: %q", chunk.Text)
				}
				if chunk.StartChar == 0 && chunk.EndChar == 0 {
					continue // normalized word joins have unknown provenance
				}
				if chunk.StartChar < 0 || chunk.EndChar <= chunk.StartChar || chunk.EndChar > len(text) || text[chunk.StartChar:chunk.EndChar] != chunk.Text {
					t.Errorf("split fence has misleading span: %#v", chunk)
				}
			}
		})
	}
}

func TestMarkdownFenceUnknownBoundaryDoesNotCrossEarlierFence(t *testing.T) {
	t.Parallel()
	text := "```\nearlier value\n```\nnot an opening fence\ncurrent value\n~~~\n"
	start := strings.Index(text, "current value")
	end := start + len("current value\n")
	gotStart, gotEnd, ok := fencedCodeSourceSpan([]byte(text), start, end)
	if ok || gotStart != 0 || gotEnd != 0 {
		t.Fatalf("unknown opening crossed earlier fence: %d:%d, found=%v", gotStart, gotEnd, ok)
	}
}

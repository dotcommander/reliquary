package chunking

import "testing"

func TestIndentedCodeWhitespaceBlankRegression(t *testing.T) {
	t.Parallel()
	for _, blank := range []string{" ", "  ", "   ", "\t", " \t "} {
		source := "    first.\n" + blank + "\n  \n    second.\nProse.\n"
		runes := []rune(source)
		ranges := findCodeBlockRanges(runes)
		if len(ranges) != 1 {
			t.Fatalf("blank %q split code: %#v", blank, ranges)
		}
		want := "    first.\n" + blank + "\n  \n    second.\n"
		got := string(runes[ranges[0].runeStart:ranges[0].runeEnd])
		if got != want {
			t.Fatalf("blank %q content = %q, want %q", blank, got, want)
		}
		spans := splitIntoSentencesWithSpans(source)
		for _, s := range spans {
			if s.end > s.start && source[s.start:s.end] != s.text {
				t.Fatalf("inexact code sentence span: %#v", s)
			}
		}
		terminated := []rune("    first.\n" + blank + "\nProse.\n    second.\n")
		ended := findIndentedBlockEnd(terminated, 0)
		if string(terminated[:ended]) != "    first.\n" {
			t.Fatalf("prose failed to terminate code: %q", string(terminated[:ended]))
		}
	}
}

func TestIndentedCodeBlankOnlyAndTrailingBlankRegression(t *testing.T) {
	t.Parallel()
	for _, blank := range []string{"   \n\t  ", "    ", "\t", "\t\n    \n  "} {
		if ranges := findCodeBlockRanges([]rune(blank)); len(ranges) != 0 {
			t.Fatalf("whitespace-only input %q became code: %#v", blank, ranges)
		}
		const code = "    猫猫.\n"
		source := code + blank
		runes := []rune(source)
		ranges := findCodeBlockRanges(runes)
		if len(ranges) != 1 || ranges[0].runeStart != 0 || ranges[0].runeEnd <= 0 {
			t.Fatalf("trailing blank %q produced invalid ranges: %#v", blank, ranges)
		}
		if got := string(runes[ranges[0].runeStart:ranges[0].runeEnd]); got != code {
			t.Fatalf("trailing blank was included in code: %q", got)
		}
	}
}

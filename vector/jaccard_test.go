package vectors

import "testing"

func TestJaccardWords_Identical(t *testing.T) {
	t.Parallel()
	s1 := "the quick brown fox jumps over the lazy dog"
	s2 := "the quick brown fox jumps over the lazy dog"
	got := JaccardWords(s1, s2)
	if got != 1.0 {
		t.Fatalf("identical strings = %f, want 1.0", got)
	}
}

func TestJaccardWords_Disjoint(t *testing.T) {
	t.Parallel()
	s1 := "cats and dogs"
	s2 := "numbers 123 and symbols"
	got := JaccardWords(s1, s2)
	if got != 0.0 {
		t.Fatalf("disjoint strings = %f, want 0.0", got)
	}
}

func TestJaccardWords_Empty(t *testing.T) {
	t.Parallel()
	s1 := ""
	s2 := ""
	got := JaccardWords(s1, s2)
	if got != 0.0 {
		t.Fatalf("both empty = %f, want 0.0", got)
	}
}

func TestJaccardWords_CorpusMatch(t *testing.T) {
	t.Parallel()
	a := "Go is fast and effective for system programming."
	b := "System programming with Go can be fast and practical."
	got := JaccardWords(a, b)
	if got <= 0.2 {
		t.Fatalf("expected some overlap, got %f", got)
	}
}

func TestJaccardWords_StripsURLsAndMathBlocks(t *testing.T) {
	t.Parallel()

	// URL containing 's' must be stripped completely, not truncated at 's'.
	if got := JaccardWords("visit https://example.com/status/docs alpha", "visit alpha"); got != 1.0 {
		t.Fatalf("URL with 's' not stripped cleanly: got %f, want 1.0", got)
	}
	if got := JaccardWords("https://example.com/status/docs", "status docs"); got != 0.0 {
		t.Fatalf("URL path tokens leaked into words: got %f, want 0.0", got)
	}
	// Words following a URL without 's' must not be swallowed by urlRE.
	if got := JaccardWords("http://example.org/path alpha beta", "alpha beta"); got != 1.0 {
		t.Fatalf("words after URL were swallowed: got %f, want 1.0", got)
	}
	// Inline/multiline $...$ math blocks must be stripped.
	if got := JaccardWords("result $alpha + beta$ gamma", "result gamma"); got != 1.0 {
		t.Fatalf("math block not stripped: got %f, want 1.0", got)
	}
	if got := JaccardWords("$alpha + beta$", "alpha beta"); got != 0.0 {
		t.Fatalf("math block tokens leaked into words: got %f, want 0.0", got)
	}
}

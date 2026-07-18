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

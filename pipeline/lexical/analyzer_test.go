package lexical

import (
	"reflect"
	"testing"
)

func TestAnalyzer_NormalizesOffsetsStopWordsAndPositions(t *testing.T) {
	t.Parallel()

	analyzer := NewAnalyzer(AnalyzerOptions{
		MinTokenLength: 2,
		StopWords: map[string]struct{}{
			"and": {},
		},
	})

	got := analyzer.Analyze("Go, Gophers! CAFÉ 42 and a β2.")
	want := []Token{
		{Text: "go", StartByte: 0, EndByte: 2, Position: 0},
		{Text: "gophers", StartByte: 4, EndByte: 11, Position: 1},
		{Text: "café", StartByte: 13, EndByte: 18, Position: 2},
		{Text: "42", StartByte: 19, EndByte: 21, Position: 3},
		{Text: "β2", StartByte: 28, EndByte: 31, Position: 4},
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Analyze() = %#v, want %#v", got, want)
	}
}

func TestAnalyzer_EmptyReturnsNonNilSlice(t *testing.T) {
	t.Parallel()

	got := NewAnalyzer(AnalyzerOptions{}).Analyze("...")
	if got == nil {
		t.Fatal("Analyze() returned nil, want non-nil empty slice")
	}
	if len(got) != 0 {
		t.Fatalf("len(Analyze()) = %d, want 0", len(got))
	}
}

func TestNormalizeQuery_RepeatedTermsAndStopwordOnly(t *testing.T) {
	t.Parallel()

	analyzer := NewAnalyzer(AnalyzerOptions{
		StopWords: map[string]struct{}{
			"the": {},
		},
	})

	got := NormalizeQuery("Alpha alpha beta", analyzer)
	if !reflect.DeepEqual(TokenTexts(got.Tokens), []string{"alpha", "alpha", "beta"}) {
		t.Fatalf("Tokens = %#v, want repeated normalized terms", got.Tokens)
	}
	if got.Terms["alpha"] != 2 || got.Terms["beta"] != 1 {
		t.Fatalf("Terms = %#v, want alpha=2 beta=1", got.Terms)
	}

	empty := NormalizeQuery("the THE", analyzer)
	if !empty.Empty() {
		t.Fatalf("stopword-only query Empty() = false, tokens=%#v terms=%#v", empty.Tokens, empty.Terms)
	}
	if empty.Tokens == nil {
		t.Fatal("stopword-only query Tokens is nil, want non-nil empty slice")
	}
}

func TestAnalyzerFunc(t *testing.T) {
	t.Parallel()

	query := NormalizeQuery("ignored", AnalyzerFunc(func(string) []Token {
		return []Token{{Text: "fixed"}}
	}))
	if query.Terms["fixed"] != 1 {
		t.Fatalf("Terms = %#v, want fixed=1", query.Terms)
	}
}

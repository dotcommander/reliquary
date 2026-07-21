package textutil

import "testing"

func TestAliasQueryScore_ToleratesPersonNameTypo(t *testing.T) {
	t.Parallel()

	got := AliasQueryScore("parent of Alex Quinn Examplf", "Alex Quinn Example", nil, nil)
	if got.Score < 0.90 {
		t.Fatalf("score = %.2f, want >= 0.90", got.Score)
	}
	if got.Reason != ReasonFuzzyToken {
		t.Fatalf("reason = %q, want %q", got.Reason, ReasonFuzzyToken)
	}
}

func TestAliasQueryScore_RejectsShortFragmentContainment(t *testing.T) {
	t.Parallel()

	got := AliasQueryScore("al", "Alex Quinn Example", nil, nil)
	if got.Score >= PersonAliasMinScore {
		t.Fatalf("score = %.2f, want below %.2f", got.Score, PersonAliasMinScore)
	}

	substring := AliasQueryScore("alexander", "Alex", nil, nil)
	if substring.Score >= PersonAliasMinScore {
		t.Fatalf("substring score = %.2f reason = %q, want below %.2f", substring.Score, substring.Reason, PersonAliasMinScore)
	}

	longSuffix := AliasQueryScore("alexson", "Alex", nil, nil)
	if longSuffix.Score >= PersonAliasMinScore {
		t.Fatalf("long-suffix score = %.2f reason = %q, want below %.2f", longSuffix.Score, longSuffix.Reason, PersonAliasMinScore)
	}
}

func TestAliasQueryScore_PhraseContainmentDirection(t *testing.T) {
	t.Parallel()

	exact := AliasQueryScore("Alex Quinn Example", "Alex Quinn Example", nil, nil)
	if exact.Score != 1 || exact.Reason != ReasonExactPhrase {
		t.Fatalf("exact match = %#v, want exact phrase", exact)
	}

	contained := AliasQueryScore("parent of Alex Quinn Example", "Alex Quinn Example", nil, nil)
	if contained.Score != PhraseContainmentScore || contained.Reason != ReasonAliasPhrase {
		t.Fatalf("contained match = %#v, want phrase containment", contained)
	}

	reversed := AliasQueryScore("alex", "Alex Quinn Example", nil, nil)
	if reversed.Score >= PersonAliasMinScore {
		t.Fatalf("reversed short containment score = %.2f, want below threshold", reversed.Score)
	}
}

func TestAliasQueryScore_AllowsMiddleNameOmission(t *testing.T) {
	t.Parallel()

	got := AliasQueryScore("Alex Example", "Alex Quinn Example", nil, nil)
	if got.Score < PersonAliasMinScore || got.Reason != ReasonTokenCoverage {
		t.Fatalf("middle-name omission match = %#v, want token coverage above threshold", got)
	}

	titleNoise := AliasQueryScore("Alex Example", "Record Alex Quinn Example", nil, nil)
	if titleNoise.Score >= PersonAliasMinScore {
		t.Fatalf("title-noise score = %.2f, want below threshold", titleNoise.Score)
	}
}

func TestAliasQueryScore_UsesDistinctTokenOccurrences(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		query     string
		canonical string
	}{
		{name: "duplicate target", query: "john", canonical: "john john"},
		{name: "exact before fuzzy", query: "anna", canonical: "anna anne"},
		{name: "exact before fuzzy reversed", query: "anna", canonical: "anne anna"},
		{name: "one token cannot cover both name boundaries", query: "anne zz", canonical: "anna middle anne"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := AliasQueryScore(tt.query, tt.canonical, nil, nil)
			if got.Score >= PersonAliasMinScore {
				t.Fatalf("AliasQueryScore(%q, %q) = %#v, want score below %.2f", tt.query, tt.canonical, got, PersonAliasMinScore)
			}
			if got.Reason != ReasonTokenCoverage {
				t.Fatalf("AliasQueryScore(%q, %q) reason = %q, want %q", tt.query, tt.canonical, got.Reason, ReasonTokenCoverage)
			}
		})
	}
}

func TestDistinctTokenMatches_UsesAugmentingPath(t *testing.T) {
	t.Parallel()

	// "marin" fuzzily matches both query terms, while "marla" only matches
	// "maria". Reaching two matches requires moving "marin" to "marie".
	hits, usedFuzzy := distinctTokenMatches(
		[]string{"maria", "marie"},
		[]string{"marin", "marla"},
	)
	if hits != 2 || !usedFuzzy {
		t.Fatalf("distinctTokenMatches() = (%d, %v), want (2, true)", hits, usedFuzzy)
	}
}

func TestAliasQueryScore_AllowsDistinctReorderedBoundaryTerms(t *testing.T) {
	t.Parallel()

	got := AliasQueryScore("Example Alex", "Alex Quinn Example", nil, nil)
	if got.Score < PersonAliasMinScore || got.Reason != ReasonTokenCoverage {
		t.Fatalf("reordered boundary match = %#v, want token coverage above threshold", got)
	}
}

func TestAliasQueryScore_CodeSymbolLookalikeStaysBelowPersonThreshold(t *testing.T) {
	t.Parallel()

	got := AliasQueryScore("derived", "$derived.by", nil, nil)
	if got.Score >= PersonAliasMinScore {
		t.Fatalf("code-symbol lookalike score = %.2f, want below %.2f", got.Score, PersonAliasMinScore)
	}
}

func TestLongTermNearMatch(t *testing.T) {
	t.Parallel()

	terms := TextTerms("Alex Example research notes")
	if !LongTermNearMatch("examplf", terms) {
		t.Fatal("expected long typo to match")
	}
	if LongTermNearMatch("probe", TextTerms("prone configuration")) {
		t.Fatal("short term unexpectedly matched fuzzily")
	}
}

func TestTermLengthThresholdsCountRunes(t *testing.T) {
	t.Parallel()

	if got := PhraseTerms("a ab \u00e9 \u00e9\u00e9 東 東京", nil); len(got) != 3 || got[0] != "ab" || got[1] != "éé" || got[2] != "東京" {
		t.Fatalf("PhraseTerms rune thresholds = %#v, want [ab éé 東京]", got)
	}
	if got := TextTerms("abc abcd \u00e9\u00e9\u00e9 \u00e9\u00e9\u00e9\u00e9 東京大 東京大阪"); len(got) != 3 || got[0] != "abcd" || got[1] != "éééé" || got[2] != "東京大阪" {
		t.Fatalf("TextTerms rune thresholds = %#v, want [abcd éééé 東京大阪]", got)
	}

	tests := []struct {
		name      string
		term      string
		candidate string
		want      bool
	}{
		{name: "six ASCII runes", term: "abcdef", candidate: "abcdeg", want: false},
		{name: "seven ASCII runes", term: "abcdefg", candidate: "abcdefh", want: true},
		{name: "six accented runes", term: "\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9", candidate: "\u00e9\u00e9\u00e9\u00e9\u00e9a", want: false},
		{name: "seven accented runes", term: "\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9", candidate: "\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9a", want: true},
		{name: "six CJK runes", term: "東京大阪京都", candidate: "東京大阪京都", want: false},
		{name: "seven CJK runes", term: "東京大阪京都駅", candidate: "東京大阪京都前", want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := LongTermNearMatch(tt.term, []string{tt.candidate}); got != tt.want {
				t.Fatalf("LongTermNearMatch(%q, [%q]) = %v, want %v", tt.term, tt.candidate, got, tt.want)
			}
		})
	}
}

func TestTermSimilarMinimumLengthCountsRunes(t *testing.T) {
	t.Parallel()

	if exact, fuzzy := termSimilar("caf\u00e9", "cafe"); exact || !fuzzy {
		t.Fatalf("termSimilar(café, cafe) = (%v, %v), want (false, true)", exact, fuzzy)
	}
	if exact, fuzzy := termSimilar("東京大", "東京小"); exact || fuzzy {
		t.Fatalf("termSimilar three-rune terms = (%v, %v), want (false, false)", exact, fuzzy)
	}
}

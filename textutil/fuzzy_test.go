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

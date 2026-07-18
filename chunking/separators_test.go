package chunking

import "testing"

func TestSeparatorProfilesReturnDetachedOrderedSeparators(t *testing.T) {
	t.Parallel()

	defaults := DefaultTextSeparatorProfile()
	if defaults.ID != SeparatorProfileDefaultTextID || defaults.Separators[0] != "\n\n" || defaults.Separators[len(defaults.Separators)-1] != "" {
		t.Fatalf("default profile = %#v", defaults)
	}

	cjk := CJKThaiSeparatorProfile()
	want := map[string]bool{"\u200b": true, "\uff0c": true, "\u3001": true, "\uff0e": true, "\u3002": true}
	for _, sep := range cjk.Separators {
		delete(want, sep)
	}
	if len(want) != 0 {
		t.Fatalf("CJK/Thai profile missing separators: %#v", want)
	}

	copied := cjk.SeparatorStrings()
	copied[0] = "mutated"
	if cjk.Separators[0] == "mutated" {
		t.Fatal("SeparatorStrings reused profile slice")
	}
}

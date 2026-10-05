package chunking

import "testing"

func TestSemanticTrimmedByteSpansRegression(t *testing.T) {
	t.Parallel()
	source := "\u2003猫猫\u2003 犬犬 \t猫猫\n"
	raw := []string{"\u2003猫猫\u2003", " 犬犬 ", "\t猫猫\n"}
	spans := locateTextSpans(source, raw)
	units := make([]SemanticUnit, len(spans))
	for i, s := range spans {
		units[i] = SemanticUnit{Text: s.text, StartChar: s.start, EndChar: s.end}
	}
	plan, ok := PlanSemanticChunks(source, units, [][]float32{{1, 0}, {0, 1}, {1, 0}}, SemanticPlanOptions{MaxChunkChars: 20, MinChunkChars: 1, FallbackSize: 20})
	if !ok {
		t.Fatal("trimmed units rejected")
	}
	for _, u := range plan.Units {
		if source[u.StartChar:u.EndChar] != u.Text {
			t.Fatalf("unit span inexact: %#v", u)
		}
	}
	assertRegressionChunkBounds(t, source, plan.Chunks, 20)
	invalid := []SemanticUnit{{Text: " 猫 ", StartChar: 1, EndChar: 8}, {Text: "犬", StartChar: 0, EndChar: 0}, {Text: "鳥", StartChar: -1, EndChar: 2}}
	normalized, _ := semanticUnitsFromPublic(invalid, source)
	for _, u := range normalized {
		if u.start != 0 || u.end != 0 {
			t.Fatalf("fabricated span %#v", u)
		}
	}
}

func TestSemanticChunkSpanTrimRegression(t *testing.T) {
	t.Parallel()
	source := "\u2003猫猫\u2003"
	chunks := buildSemanticChunks([]textSpan{{text: "猫猫", start: 0, end: len(source)}}, []string{"猫猫"}, source)
	assertRegressionChunkBounds(t, source, chunks, 2)
	if chunks[0].StartChar != len("\u2003") {
		t.Fatalf("lost trim offset: %#v", chunks)
	}
}

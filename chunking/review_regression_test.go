package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSplitIntoChunksNonpositiveLength(t *testing.T) {
	t.Parallel()
	for _, length := range []int{0, -1} {
		c := OptimalChunker{OptimalLength: length}
		if got := c.SplitIntoChunks("hello world"); len(got) != 0 {
			t.Fatalf("SplitIntoChunks with OptimalLength %d = %v, want empty", length, got)
		}
	}
	var zero OptimalChunker
	if got := zero.SplitIntoChunks("hello world"); len(got) != 0 {
		t.Fatalf("zero-value SplitIntoChunks = %v, want empty", got)
	}
}

func TestPlanSemanticChunksEnforcesDefaultCeiling(t *testing.T) {
	t.Parallel()
	paragraph := strings.Repeat("word ", 400)
	source := paragraph + "\n\n" + paragraph + "\n\n" + paragraph
	units := SemanticUnits(source)
	if len(units) < 3 {
		t.Fatalf("SemanticUnits = %d units, want at least 3", len(units))
	}
	embeddings := make([][]float32, len(units))
	for i := range embeddings {
		embeddings[i] = []float32{1, 0}
	}
	plan, ok := PlanSemanticChunks(source, units, embeddings, SemanticPlanOptions{MinChunkChars: 1})
	if !ok {
		t.Fatal("valid plan rejected")
	}
	for _, c := range plan.Chunks {
		if n := utf8.RuneCountInString(c.Text); n > defaultMaxChunkChars {
			t.Fatalf("chunk has %d runes, exceeds default ceiling %d", n, defaultMaxChunkChars)
		}
	}
}

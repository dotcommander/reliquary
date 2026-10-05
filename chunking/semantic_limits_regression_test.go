package chunking

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"
)

type identicalRegressionEmbedder struct{}

func (identicalRegressionEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range result {
		result[i] = []float32{1, 0}
	}
	return result, nil
}

func TestSemanticMergeCeilingsRegression(t *testing.T) {
	t.Parallel()
	source := "First sentence has enough content. Second sentence has enough content. Third sentence has enough content."
	units := SemanticUnits(source)
	embeddings, _ := (identicalRegressionEmbedder{}).EmbedBatch(t.Context(), make([]string, len(units)))
	for _, controls := range []struct{ max, fallback int }{{40, 80}, {80, 40}, {0, 0}, {0, 40}, {40, 0}} {
		plan, ok := PlanSemanticChunks(source, units, embeddings, SemanticPlanOptions{MaxChunkChars: controls.max, MinChunkChars: 1, FallbackSize: controls.fallback})
		if !ok {
			t.Fatal("valid plan rejected")
		}
		size := controls.max
		if size <= 0 {
			size = defaultMaxChunkChars
		}
		if controls.fallback > 0 && controls.fallback < size {
			size = controls.fallback
		}
		assertRegressionChunkBounds(t, source, plan.Chunks, size)
		sc, err := NewSemanticChunker(identicalRegressionEmbedder{}, SemanticOpts{MaxChunkChars: controls.max, MinChunkChars: 1})
		if err != nil {
			t.Fatal(err)
		}
		assertRegressionChunkBounds(t, source, sc.ChunkSemantic(t.Context(), source, controls.fallback, 0), size)
	}
}

func TestSemanticAdjacentMergeLimitRegression(t *testing.T) {
	t.Parallel()
	groups := []semanticGroup{{text: "猫猫", embedding: []float32{1}, weight: 2}, {text: "犬犬", embedding: []float32{1}, weight: 2}, {text: "鳥鳥", embedding: []float32{1}, weight: 2}}
	bounded := mergeAdjacentSimilarGroups(groups, 0.9, 5)
	if len(bounded) != 2 {
		t.Fatalf("bounded merge = %#v", bounded)
	}
	for _, g := range bounded {
		if utf8.RuneCountInString(g.text) > 5 {
			t.Fatal("merge exceeded configured maximum")
		}
	}
	for _, limit := range []int{0, 20} {
		merged := mergeAdjacentSimilarGroups(groups, 0.9, limit)
		if len(merged) != 1 || !strings.Contains(merged[0].text, "鳥鳥") {
			t.Fatalf("default/control merge = %#v", merged)
		}
	}
}

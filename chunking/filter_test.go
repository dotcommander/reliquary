package chunking

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFilterProse_SkipsCode(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 0, Text: "Some meaningful prose here.", Metadata: map[string]string{"type": "paragraph", metaKeyWordCount: "5"}},
		{ID: 1, Text: "func main() {}", Metadata: map[string]string{"type": "code", metaKeyLanguage: "go"}},
	}
	result := FilterProse(chunks)
	assert.Len(t, result, 1)
	assert.Equal(t, 0, result[0].ID)
}

func TestFilterProse_SkipsTable(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 0, Text: "Some meaningful prose here.", Metadata: map[string]string{"type": "paragraph", metaKeyWordCount: "5"}},
		{ID: 1, Text: "| A | B |", Metadata: map[string]string{"type": "table"}},
	}
	result := FilterProse(chunks)
	assert.Len(t, result, 1)
	assert.Equal(t, 0, result[0].ID)
}

func TestFilterProse_DropsShortParagraph(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 0, Text: "Hi.", Metadata: map[string]string{"type": "paragraph", metaKeyWordCount: "1"}},
		{ID: 1, Text: "This is a longer paragraph with enough words to pass.", Metadata: map[string]string{"type": "paragraph", metaKeyWordCount: "10"}},
	}
	result := FilterProse(chunks)
	assert.Len(t, result, 1)
	assert.Equal(t, 1, result[0].ID)
}

func TestFilterProse_AdmitsHeadingAtFloor(t *testing.T) {
	t.Parallel()

	// Heading with exactly 3 words (HeadingWordFloor) — should be admitted.
	chunks := []Chunk{
		{ID: 0, Text: "# Three Word Title", Metadata: map[string]string{"type": "heading", metaKeyHeadingLevel: "1", metaKeyWordCount: "3"}},
		{ID: 1, Text: "## Short", Metadata: map[string]string{"type": "heading", metaKeyHeadingLevel: "2", metaKeyWordCount: "1"}},
	}
	result := FilterProse(chunks)
	assert.Len(t, result, 1)
	assert.Equal(t, 0, result[0].ID)
}

func TestFilterProse_NilMetadataPassThrough(t *testing.T) {
	t.Parallel()

	chunks := []Chunk{
		{ID: 0, Text: "Unknown chunk"},
		{ID: 1, Text: "Another one"},
	}
	result := FilterProse(chunks)
	assert.Len(t, result, 2)
}

func TestFilterProse_EmptyInput(t *testing.T) {
	t.Parallel()

	result := FilterProse(nil)
	assert.Nil(t, result)

	result = FilterProse([]Chunk{})
	assert.Empty(t, result)
}

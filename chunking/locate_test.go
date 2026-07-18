package chunking

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLocate_ExactMatch(t *testing.T) {
	t.Parallel()

	content := "hello world foo bar"
	start, end, ok := Locate(content, "foo", 0)
	assert.True(t, ok)
	assert.Equal(t, 12, start)
	assert.Equal(t, 15, end)
	assert.Equal(t, "foo", content[start:end])
}

func TestLocate_ExactMatchFromCursor(t *testing.T) {
	t.Parallel()

	content := "aaa bbb aaa bbb"
	// Second "aaa" starting from cursor 4.
	start, end, ok := Locate(content, "aaa", 4)
	assert.True(t, ok)
	assert.Equal(t, 8, start)
	assert.Equal(t, 11, end)
	assert.Equal(t, "aaa", content[start:end])
}

func TestLocate_ExactMatchFromZero(t *testing.T) {
	t.Parallel()

	content := "alpha beta gamma"
	// Fragment "alpha" exists at 0 but cursor is past it; should find from zero.
	start, end, ok := Locate(content, "alpha", 5)
	assert.True(t, ok)
	assert.Equal(t, 0, start)
	assert.Equal(t, 5, end)
}

func TestLocate_NormalizedMatch(t *testing.T) {
	t.Parallel()

	// Fragment has collapsed whitespace; content has multiple spaces/newlines.
	content := "hello   world\n\nfoo"
	start, end, ok := Locate(content, "hello world", 0)
	assert.True(t, ok)
	assert.Equal(t, "hello   world", content[start:end])
}

func TestLocate_EmptyFragment(t *testing.T) {
	t.Parallel()

	start, end, ok := Locate("content", "", 0)
	assert.False(t, ok)
	assert.Equal(t, 0, start)
	assert.Equal(t, 0, end)
}

func TestLocate_NotFound(t *testing.T) {
	t.Parallel()

	start, end, ok := Locate("hello world", "missing", 0)
	assert.False(t, ok)
	assert.Equal(t, 0, start)
	assert.Equal(t, 0, end)
}

func TestLocate_MultibyteContent(t *testing.T) {
	t.Parallel()

	// Chinese characters: each rune is 3 bytes.
	content := "你好世界。再见。"
	start, end, ok := Locate(content, "再见", 0)
	assert.True(t, ok)
	assert.Equal(t, "再见", content[start:end])
	// Verify byte offsets are valid.
	assert.Equal(t, "再见", content[start:end])
}

func TestLocate_CursorBeyondEnd(t *testing.T) {
	t.Parallel()

	content := "hello world"
	start, end, ok := Locate(content, "hello", 100)
	assert.True(t, ok)
	assert.Equal(t, 0, start)
	assert.Equal(t, 5, end)
}

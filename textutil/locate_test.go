package textutil

import (
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFragmentRange(t *testing.T) {
	t.Parallel()

	t.Run("exact_match", func(t *testing.T) {
		t.Parallel()
		content := "hello world foo"
		fragment := "world"
		start, end, found := FragmentRange(content, fragment, 0, NormalizedEarly)
		require.True(t, found)
		assert.Equal(t, fragment, content[start:end])
	})

	t.Run("empty_fragment_returns_not_found", func(t *testing.T) {
		t.Parallel()
		start, end, found := FragmentRange("hello world", "", 0, NormalizedEarly)
		assert.False(t, found)
		assert.Equal(t, 0, start)
		assert.Equal(t, 0, end)
	})

	t.Run("negative_cursor_clamped_no_panic", func(t *testing.T) {
		t.Parallel()
		content := "hello world"
		fragment := "hello"
		start, end, found := FragmentRange(content, fragment, -5, NormalizedEarly)
		require.True(t, found)
		assert.Equal(t, fragment, content[start:end])
	})

	t.Run("cursor_past_end_clamped", func(t *testing.T) {
		t.Parallel()
		content := "hello world"
		fragment := "hello"
		start, end, found := FragmentRange(content, fragment, len(content)+100, NormalizedEarly)
		require.True(t, found)
		assert.Equal(t, fragment, content[start:end])
	})

	t.Run("cursor_skips_earlier_occurrence", func(t *testing.T) {
		t.Parallel()
		content := "foo bar foo baz"
		fragment := "foo"
		// cursor positioned after the first "foo"
		start, end, found := FragmentRange(content, fragment, 4, NormalizedEarly)
		require.True(t, found)
		// should find the second "foo" at byte 8
		assert.Equal(t, 8, start)
		assert.Equal(t, fragment, content[start:end])
	})

	t.Run("cursor_prefers_later_normalized_match_before_earlier_exact_fallback", func(t *testing.T) {
		t.Parallel()
		content := "a b\nx\na\nb"
		start, end, found := FragmentRange(content, "a b", 4, NormalizedEarly)
		require.True(t, found)
		assert.Equal(t, "a\nb", content[start:end])
	})

	t.Run("not_found", func(t *testing.T) {
		t.Parallel()
		_, _, found := FragmentRange("hello world", "nothere", 0, NormalizedEarly)
		assert.False(t, found)
	})

	t.Run("whitespace_normalized_fallback", func(t *testing.T) {
		t.Parallel()
		content := "hello\n   world"
		fragment := "hello world"
		start, end, found := FragmentRange(content, fragment, 0, NormalizedEarly)
		require.True(t, found)
		// returned slice must be valid UTF-8 and contain "hello"
		assert.True(t, utf8.ValidString(content[start:end]))
		assert.Contains(t, content[start:end], "hello")
	})

	t.Run("multibyte_exact", func(t *testing.T) {
		t.Parallel()
		content := "say wörld"
		fragment := "wörld"
		start, end, found := FragmentRange(content, fragment, 0, NormalizedEarly)
		require.True(t, found)
		assert.Equal(t, fragment, content[start:end])
	})

	t.Run("multibyte_normalized_fallback", func(t *testing.T) {
		t.Parallel()
		content := "say\n  wörld"
		fragment := "say wörld"
		start, end, found := FragmentRange(content, fragment, 0, NormalizedEarly)
		require.True(t, found)
		assert.True(t, utf8.ValidString(content[start:end]))
	})

	t.Run("malformed_utf8_normalized_fallback_is_slice_safe", func(t *testing.T) {
		t.Parallel()
		content := "prefix\n\xff"
		fragment := "prefix \ufffd"
		start, end, found := FragmentRange(content, fragment, 0, NormalizedEarly)
		require.True(t, found)
		assert.GreaterOrEqual(t, start, 0)
		assert.GreaterOrEqual(t, end, start)
		assert.LessOrEqual(t, end, len(content))
		assert.Equal(t, content, content[start:end])
	})
}

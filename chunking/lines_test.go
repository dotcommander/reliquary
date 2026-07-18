package chunking

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLineForOffset_FirstLine(t *testing.T) {
	t.Parallel()
	assert.Equal(t, 1, LineForOffset("abc\ndef", 0))
}

func TestLineForOffset_SecondLine(t *testing.T) {
	t.Parallel()
	assert.Equal(t, 2, LineForOffset("abc\ndef", 4))
}

func TestLineForOffset_ExactNewline(t *testing.T) {
	t.Parallel()
	// Offset at the \n itself is still line 1.
	assert.Equal(t, 1, LineForOffset("a\nb", 1))
}

func TestLineForOffset_NegativeOffset(t *testing.T) {
	t.Parallel()
	assert.Equal(t, 1, LineForOffset("abc", -5))
}

func TestLineForOffset_OffsetBeyondEnd(t *testing.T) {
	t.Parallel()
	assert.Equal(t, 2, LineForOffset("a\nb", 100))
}

func TestLineForOffset_EmptyContent(t *testing.T) {
	t.Parallel()
	assert.Equal(t, 1, LineForOffset("", 0))
}

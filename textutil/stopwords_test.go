package textutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestIsStopWord_DoesNotMutate verifies that IsStopWord is read-only.
func TestIsStopWord_DoesNotMutate(t *testing.T) {
	t.Parallel()

	before := len(DefaultStopWordsCopy())

	assert.True(t, IsStopWord("the"))
	assert.False(t, IsStopWord("exampleperson"))

	assert.Equal(t, before, len(DefaultStopWordsCopy()), "IsStopWord must not modify default stop words")
	assert.False(t, IsStopWord("exampleperson"), "absent word must remain absent after lookup")
}

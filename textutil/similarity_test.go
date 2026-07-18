package textutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStringSimilarity_KnownValues(t *testing.T) {
	t.Parallel()

	t.Run("martha_marhta", func(t *testing.T) {
		t.Parallel()
		assert.InDelta(t, 0.9611, StringSimilarity("martha", "marhta"), 0.001)
	})

	t.Run("dwayne_duane", func(t *testing.T) {
		t.Parallel()
		assert.InDelta(t, 0.8400, StringSimilarity("dwayne", "duane"), 0.001)
	})

	t.Run("dixon_dicksonx", func(t *testing.T) {
		t.Parallel()
		assert.InDelta(t, 0.8133, StringSimilarity("dixon", "dicksonx"), 0.001)
	})

	t.Run("abc_xyz_zero", func(t *testing.T) {
		t.Parallel()
		assert.InDelta(t, 0.0, StringSimilarity("abc", "xyz"), 0.001)
	})

	t.Run("same_word", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, 1.0, StringSimilarity("alex", "alex"))
	})

	t.Run("both_empty", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, 1.0, StringSimilarity("", ""))
	})

	t.Run("second_empty", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, 0.0, StringSimilarity("alex", ""))
	})

	t.Run("case_insensitive", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, 1.0, StringSimilarity("Alex", "alex"))
	})
}

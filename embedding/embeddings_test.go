package embeddings

import "testing"

func TestValidateDimensions(t *testing.T) {
	t.Parallel()

	err := ValidateDimensions([]Vector{{1, 2}, {3}}, 2)
	if err == nil {
		t.Fatal("expected dimension error")
	}
}

func TestCacheKeyIncludesModel(t *testing.T) {
	t.Parallel()

	input := "same text"
	a := CacheKey(ModelRef{Provider: "local", Name: "a", Dim: 2}, input)
	b := CacheKey(ModelRef{Provider: "local", Name: "b", Dim: 2}, input)
	if a == b {
		t.Fatal("cache key should include model identity")
	}
}

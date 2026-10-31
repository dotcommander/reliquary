package retrieval

import (
	"slices"
	"testing"
)

func TestFormatContextUnboundedKeepsBlocksAndSummary(t *testing.T) {
	t.Parallel()
	results := []*Result{{Content: "one"}, nil, {Content: "two"}}
	got, summary, err := FormatContextWithSummary(results, WithSeparator(" | "), WithContentTransform(func(item ContextItem) (string, error) {
		return item.Content + "!", nil
	}))
	if err != nil || got != "one! | two!" || !slices.Equal(summary.IncludedIndices, []int{0, 2}) || !slices.Equal(summary.SkippedIndices, []int{1}) || summary.TokensUsed != nil {
		t.Fatalf("unbounded context = %q, %+v, %v", got, summary, err)
	}
}

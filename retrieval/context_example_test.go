package retrieval_test

import (
	"fmt"
	"strings"

	"github.com/dotcommander/reliquary/retrieval"
)

type wordCounter struct{}

func (wordCounter) Count(text string) (int, error) {
	return len(strings.Fields(text)), nil
}

func ExampleFormatContext() {
	results := []*retrieval.Result{
		{
			Filename: "gc.md",
			Content:  "Go uses a concurrent garbage collector.",
			Metadata: map[string]any{
				retrieval.ContextStartLineKey: 12,
				retrieval.ContextEndLineKey:   12,
			},
		},
		{
			DocumentID: "scheduler",
			Content:    "Goroutines are multiplexed onto threads.",
			Metadata: map[string]any{
				retrieval.ContextStartLineKey: 4,
				retrieval.ContextEndLineKey:   5,
			},
		},
	}

	promptBlock, err := retrieval.FormatContext(results,
		retrieval.WithHeader("[Source: %s, Lines: %d-%d]"),
		retrieval.WithMaxTokens(2048, wordCounter{}),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(promptBlock)
	// Output:
	// [Source: gc.md, Lines: 12-12]
	// Go uses a concurrent garbage collector.
	//
	// [Source: scheduler, Lines: 4-5]
	// Goroutines are multiplexed onto threads.
}

func ExampleFormatContextWithSummary() {
	results := []*retrieval.Result{
		{Filename: "notes.md", Content: "  First passage  ", Metadata: map[string]any{"title": "Notes"}},
		{Content: ""},
		{Filename: "other.md", Content: "Second passage", Metadata: map[string]any{"title": "Other"}},
	}
	text, summary, err := retrieval.FormatContextWithSummary(results,
		retrieval.WithHeaderTemplate("{{.Result.Metadata.title}} — {{.Source}}"),
		retrieval.WithCitationTemplate("[{{.Number}}]"),
		retrieval.WithContentTransform(func(item retrieval.ContextItem) (string, error) { return strings.TrimSpace(item.Content), nil }),
		retrieval.WithMaxTokens(6, wordCounter{}),
	)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(text)
	fmt.Println("included:", summary.IncludedIndices, "skipped:", summary.SkippedIndices, "omitted:", summary.OmittedIndices)
	// Output:
	// [1] Notes — notes.md
	// First passage
	// included: [0] skipped: [1] omitted: [2]
}

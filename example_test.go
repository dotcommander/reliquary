package reliquary_test

import (
	"context"
	"fmt"

	"github.com/dotcommander/reliquary"
	"github.com/dotcommander/reliquary/document"
)

func ExampleQuickstart() {
	app := reliquary.Quickstart()
	ctx := context.Background()
	_, _ = app.Ingest(ctx, document.Document{
		ID:   "doc-1",
		Text: "Go uses a concurrent garbage collector.",
	})

	hits, _ := app.Search(ctx, "garbage collector", reliquary.TopK(1))
	fmt.Println(len(hits), hits[0].ID != "")
	// Output: 1 true
}

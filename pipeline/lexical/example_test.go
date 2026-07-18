package lexical_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/pipeline/lexical"
)

func ExampleRankByOrder() {
	ranked := lexical.RankByOrder([]lexical.Result{
		{ID: "doc-a"},
		{ID: "doc-b"},
	}, "sqlite_fts5", lexical.ScoreSpaceRankOnly)
	fmt.Println(lexical.RankedIDs(ranked))
	// Output: [doc-a doc-b]
}

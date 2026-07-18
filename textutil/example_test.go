package textutil_test

import (
	"fmt"

	"github.com/dotcommander/reliquary/textutil"
)

func ExampleExtractKeywords() {
	keywords := textutil.ExtractKeywords(
		[]string{
			"Go routines and channels make concurrency practical.",
			"Channels help coordinate goroutines in Go.",
		},
		textutil.KeywordOptions{Limit: 3, MinCount: 1},
	)
	fmt.Println(keywords)
	// Output:
	// [channels concurrency coordinate]
}

func ExampleExtractKeywords_perCallStopWords() {
	keywords := textutil.ExtractKeywords(
		[]string{
			"Channels help coordinate channels in Go.",
			"Goroutines coordinate work through channels.",
		},
		textutil.KeywordOptions{
			Limit:     3,
			MinCount:  1,
			StopWords: map[string]struct{}{"channels": {}},
		},
	)
	fmt.Println(keywords)
	// Output:
	// [coordinate goroutines help]
}

func ExampleAliasQueryScore() {
	match := textutil.AliasQueryScore(
		"parent of Alex Quinn Examplf",
		"Alex Quinn Example",
		nil,
		nil,
	)
	fmt.Printf("%.2f %s\n", match.Score, match.Reason)
	// Output:
	// 1.00 fuzzy_token
}

func ExampleFragmentRange() {
	content := "hello\n   world"
	start, end, found := textutil.FragmentRange(content, "hello world", 0, textutil.NormalizedEarly)
	fmt.Println(found)
	fmt.Println(content[start:end])
	// Output:
	// true
	// hello
	//    world
}

func ExampleMostFrequentValue() {
	// Hit: "docs" appears twice, meeting minCount 2.
	fmt.Println(textutil.MostFrequentValue([]string{"docs", "api", "docs"}, 2, "fallback"))

	// Fallback: no value reaches minCount 2.
	fmt.Println(textutil.MostFrequentValue([]string{"docs", "api"}, 2, "fallback"))
	// Output:
	// docs
	// fallback
}

func ExampleDetectTheme() {
	themes := map[string][]string{
		"engineering": {"go", "chan", "channel"},
		"docs":        {"readme", "guide"},
	}

	// Match: "Go" scores for "engineering"; "channel" no longer matches "channels".
	fmt.Println(textutil.DetectTheme("This project uses Go channels heavily.", themes, "fallback"))

	// Fallback: no theme keyword appears in the content.
	fmt.Println(textutil.DetectTheme("No related language appears", themes, "fallback"))
	// Output:
	// engineering
	// fallback
}

func ExampleNormalizeKeywordToken() {
	fmt.Println(textutil.NormalizeKeywordToken("(channels)"))
	fmt.Println(textutil.NormalizeKeywordToken("go."))
	// Output:
	// channels
	// go
}

func ExampleIsStopWord() {
	fmt.Println(textutil.IsStopWord("the"))
	fmt.Println(textutil.IsStopWord("channels"))
	// Output:
	// true
	// false
}

func ExampleDefaultStopWordsCopy() {
	words := textutil.DefaultStopWordsCopy()
	words["channels"] = struct{}{}

	_, inCopy := words["channels"]
	fmt.Println(inCopy)
	fmt.Println(textutil.IsStopWord("channels"))
	// Output:
	// true
	// false
}

func ExampleTitleWords() {
	fmt.Println(textutil.TitleWords("hello_world-foo"))
	// Output:
	// Hello World Foo
}

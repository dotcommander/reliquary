# textutil

Stdlib-only text helpers for keyword extraction, theme detection, fuzzy alias matching, fragment location, and title normalization.

```go
import "github.com/dotcommander/reliquary/textutil"
```

```go
keywords := textutil.ExtractKeywords([]string{
    "Go routines and channels make concurrency practical.",
    "Channels help coordinate goroutines in Go.",
}, textutil.KeywordOptions{Limit: 3, MinCount: 1})
// → []string{"channels", "concurrency", "coordinate"}

textutil.TitleWords("hello_world-foo")
// → "Hello World Foo"

match := textutil.AliasQueryScore("father of Jordan Morgan", "Jordan Morgan", nil, nil)
// → textutil.AliasMatch{Score: 1, Reason: "fuzzy_token"}
```

## Install

```sh
go get github.com/dotcommander/reliquary/textutil
```

## What it does

`textutil` provides three categories of helpers:

- **Keyword extraction** — `ExtractKeywords` tokenizes a slice of texts, strips punctuation and stop words, and returns the top tokens by frequency.
- **Theme detection** — `DetectTheme` scores text against named keyword lists and returns the best-matching theme name.
- **Fuzzy matching** — `AliasQueryScore` scores person-name style query/canonical/alias matches and reports why the best match won.
- **Fragment location** — `FragmentRange` returns byte offsets for exact or whitespace-normalized fragment matches.
- **Title normalization** — `TitleWords` converts slug-style identifiers (`"hello_world-foo"`) into human-readable labels (`"Hello World Foo"`).

Supporting primitives: `MostFrequentValue`, `NormalizeKeywordToken`, `IsStopWord`, `DefaultStopWords`, `DefaultStopWordsCopy`, `StringSimilarity`, `PhraseTerms`, `TextTerms`, `LongTermNearMatch`.

`DefaultStopWords` is a compatibility snapshot for callers that use the old map-shaped API. `DefaultStopWordsCopy()` returns a fresh copy. Add domain-specific stop words with `KeywordOptions.StopWords` so concurrent callers do not share mutable global state.

See [API reference](docs/api.md) for every function, option, and example.

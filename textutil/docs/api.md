# textutil

```go
import "github.com/dotcommander/reliquary/textutil"
```

```go
keywords := textutil.ExtractKeywords([]string{
    "Go routines and channels make concurrency practical.",
    "Channels help coordinate goroutines in Go.",
}, textutil.KeywordOptions{Limit: 3, MinCount: 1})
// → []string{"channels", "concurrency", "coordinate"}

theme := textutil.DetectTheme("This project uses Go channels heavily.", map[string][]string{
    "engineering": {"go", "chan", "channel"},
    "docs":        {"readme", "guide"},
}, "fallback")
// → "engineering"

label := textutil.TitleWords("hello_world-foo")
// → "Hello World Foo"

match := textutil.AliasQueryScore("father of Jordan Morgan", "Jordan Morgan", nil, nil)
// → textutil.AliasMatch{Score: 1, Reason: "fuzzy_token"}
```

---

## extracting-keywords

`ExtractKeywords` accepts a slice of texts and a `KeywordOptions` config. It tokenizes each text by whitespace, lowercases every token, strips leading/trailing punctuation via `NormalizeKeywordToken`, drops stop words and tokens below `MinLength`, counts occurrences, and returns up to `Limit` tokens sorted by descending frequency. Ties break alphabetically.

```go
keywords := textutil.ExtractKeywords([]string{
    "Go routines and channels make concurrency practical.",
    "Channels help coordinate goroutines in Go.",
}, textutil.KeywordOptions{Limit: 3, MinCount: 1})
// → []string{"channels", "concurrency", "coordinate"}
```

### keyword-options

Configure extraction behavior with `KeywordOptions`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Limit` | `int` | `10` | Maximum number of keywords to return. |
| `MinLength` | `int` | `4` | Minimum character length a token must have after normalization. |
| `MinCount` | `int` | `0` | Minimum number of occurrences a token must appear across all texts. A value of `0` means no minimum (every token that appears at least once qualifies). |
| `StopWords` | `map[string]struct{}` | `nil` | Optional per-call stop-word set layered on top of the package defaults. Use this for domain words without mutating global state. |
| `Include` | `func(string) bool` | `nil` | Optional predicate applied after stop-word and length filtering. Return `false` to exclude a token. `nil` accepts all. |

Defaults apply when a field is zero: `Limit` and `MinLength` fall back to `10` and `4` respectively. `MinCount` has no implicit default — `0` means every qualifying token passes.

```go
// Accept only tokens that contain a vowel
keywords := textutil.ExtractKeywords(texts, textutil.KeywordOptions{
	Limit:     5,
	MinLength: 3,
	MinCount:  2,
	StopWords: map[string]struct{}{
		"channels": {},
	},
	Include: func(token string) bool {
		return strings.ContainsAny(token, "aeiou")
	},
})
```

---

## detecting-a-theme

`DetectTheme` scores `content` against each theme by counting case-insensitive whole-word hits for every keyword in the theme's list (content is split on whitespace; a keyword matches only a full token, not a substring inside a larger word). It returns the name of the highest-scoring theme, or `fallback` when all themes score zero.

```go
themes := map[string][]string{
    "engineering": {"go", "chan", "channel"},
    "docs":        {"readme", "guide"},
}

textutil.DetectTheme("This project uses Go channels heavily.", themes, "fallback")
// → "engineering"

textutil.DetectTheme("No related language appears", themes, "fallback")
// → "fallback"
```

Themes are sorted alphabetically before scoring, so when two themes reach the same score the alphabetically earlier name wins. Build your keyword lists and theme names with this tie-breaking rule in mind.

---

## matching-aliases

`AliasQueryScore` scores a query against a canonical phrase and optional aliases. It is tuned for person-name style matching, not arbitrary code-symbol or substring search. It returns an `AliasMatch` with a numeric score and a `MatchReason` explaining the best match.

```go
match := textutil.AliasQueryScore(
	"father of Jordan Morgan",
	"Jordan Morgan",
	nil,
	nil,
)
// → textutil.AliasMatch{Score: 1, Reason: textutil.ReasonFuzzyToken}
```

Use `PersonAliasMinScore` (`0.80`) as the package threshold for person alias matches:

```go
if match.Score >= textutil.PersonAliasMinScore {
	// Treat the query as matching the canonical person name.
}
```

### alias-match-reasons

`AliasMatch.Reason` is one of:

| Reason | Meaning |
|--------|---------|
| `ReasonExactPhrase` | Query terms exactly match the canonical phrase. |
| `ReasonAliasPhrase` | The canonical phrase or an alias appears as a whole term sequence inside the query. |
| `ReasonTokenCoverage` | Query terms cover enough target terms, including first/last name coverage for longer names. |
| `ReasonFuzzyToken` | A long enough token matched fuzzily via `StringSimilarity`. |
| `ReasonNone` | No match was found. |

### phrase-and-text-terms

`PhraseTerms` lowercases text, splits on non-letter/non-digit runes, drops one-character terms, and applies an optional `StopTermFunc`.

```go
terms := textutil.PhraseTerms("Parent of Jordan Morgan", textutil.IsStopWord)
// → []string{"parent", "jordan", "morgan"}
```

`TextTerms` lowercases text, splits on non-letter/non-digit runes, keeps unique terms of length four or greater, and preserves first-seen order. `LongTermNearMatch` checks whether a long term has a bounded fuzzy match inside a `TextTerms` result.

```go
terms := textutil.TextTerms("Jordan Morgan research notes")
textutil.LongTermNearMatch("morgn", terms)
// → true
```

### string-similarity

`StringSimilarity` returns the case-insensitive Jaro-Winkler similarity of two strings as a number in `[0, 1]`. It is stdlib-only and intentionally matches the previous `github.com/adrg/strutil/metrics` Jaro-Winkler behavior used by this package.

```go
textutil.StringSimilarity("martha", "marhta")
// → 0.961...

textutil.StringSimilarity("Jordan", "jordan")
// → 1
```

---

## locating-fragments

`FragmentRange` locates a fragment inside content and returns byte offsets `(start, end, found)`. It first searches for an exact match at or after `cursor`, then tries a whitespace-normalized match at or after `cursor`, then falls back to exact and normalized matches before the cursor. Negative or past-end cursors are clamped to `0`.

```go
content := "hello\n   world"
start, end, found := textutil.FragmentRange(content, "hello world", 0)
// found → true
// content[start:end] → "hello\n   world"
```

Offsets are byte offsets into the original content, so the returned range can be used directly for slicing or highlighting. Normalized matches preserve valid UTF-8 ranges for multibyte text.

---

## finding-the-most-frequent-value

`MostFrequentValue` scans a string slice, trims whitespace from each entry, skips empty strings, and returns the value that appears most often. If no value meets `minCount`, it returns `fallback`.

```go
textutil.MostFrequentValue([]string{"docs", "api", "docs"}, 2, "fallback")
// → "docs"

textutil.MostFrequentValue([]string{"docs", "api"}, 2, "fallback")
// → "fallback"
```

Pass `minCount: 1` to return the most frequent value regardless of how many times it appears.

---

## normalizing-titles

`TitleWords` replaces `_` and `-` with spaces, then title-cases each word.

```go
textutil.TitleWords("hello_world-foo")
// → "Hello World Foo"
```

Use it to convert slug-style identifiers into human-readable labels for display.

---

## customizing-stop-words

### defaultstopwords

`DefaultStopWords` is a compatibility snapshot of the package default stop-word set containing 69 common English words (`"the"`, `"and"`, `"or"`, etc.). It preserves the historical map-shaped API for callers that index, range, or pass the default set.

Package behavior reads an internal default set, so mutating `DefaultStopWords` never changes `IsStopWord` or `ExtractKeywords`.

`DefaultStopWordsCopy()` returns a fresh copy of the package defaults.

Use `KeywordOptions.StopWords` to add per-call domain stop words, or `KeywordOptions.Include` for custom predicates.

```go
keywords := textutil.ExtractKeywords(texts, textutil.KeywordOptions{
	Limit: 5,
	StopWords: map[string]struct{}{
		"channels": {},
	},
})
```

```go
words := textutil.DefaultStopWordsCopy()
words["channels"] = struct{}{}
textutil.IsStopWord("channels") // → false; the copy does not alter defaults
```

### isstopword

`IsStopWord` reports whether a word is present in the package default stop-word set.

```go
textutil.IsStopWord("the")      // → true
textutil.IsStopWord("channels") // → false
```

### normalizekeywordtoken

`NormalizeKeywordToken` strips leading and trailing punctuation — `. , ; : ( ) [ ] { } " ' \` ! ? < >` — from a single token. `ExtractKeywords` calls it internally on every whitespace-split token before filtering.

```go
textutil.NormalizeKeywordToken("(channels)")  // → "channels"
textutil.NormalizeKeywordToken("go.")         // → "go"
textutil.NormalizeKeywordToken("\"quoted\"")  // → "quoted"
```

Call it directly when you need to normalize tokens outside the `ExtractKeywords` pipeline.

# dedup

[![Go Reference](https://pkg.go.dev/badge/github.com/dotcommander/reliquary/dedup.svg)](https://pkg.go.dev/github.com/dotcommander/reliquary/dedup)
[![Go Report Card](https://goreportcard.com/badge/github.com/dotcommander/reliquary/dedup)](https://goreportcard.com/report/github.com/dotcommander/reliquary/dedup)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Generic, dependency-free duplicate and near-duplicate detection for Go.

Point it at any slice, tell it how to read the text out of each item, and it
groups the duplicates for you. Exact matches via content hashing; fuzzy matches
via [SimHash](https://en.wikipedia.org/wiki/SimHash) and Hamming distance. No
database, no external services — just the standard library and generics.

```go
d := dedup.NewDetector[Doc](dedup.NormalizedHash, func(x Doc) string { return x.Body })
d.Index(docs)
groups := d.FindDuplicates() // [][]Doc, each inner slice is one set of duplicates
```

## Install

```sh
go get github.com/dotcommander/reliquary/dedup
```

Requires Go 1.26+. The only non-stdlib dependency is `testify`, used by tests.

## Quick start

Find exact duplicates in a slice of your own type:

```go
package main

import (
	"fmt"

	"github.com/dotcommander/reliquary/dedup"
)

type Doc struct {
	ID, Body string
}

func main() {
	docs := []Doc{
		{ID: "a", Body: "the quick brown fox"},
		{ID: "b", Body: "the quick brown fox"},
		{ID: "c", Body: "something else"},
	}

	d := dedup.NewDetector[Doc](dedup.SimpleHash, func(x Doc) string { return x.Body })
	d.Index(docs)

	for _, group := range d.FindDuplicates() {
		fmt.Printf("%d copies: %s\n", len(group), group[0].Body)
	}
	// 2 copies: the quick brown fox
}
```

## Near-duplicates

`SimHash` is the only strategy that supports fuzzy matching. Index with it, then
call `FindNearDuplicates` with a Hamming-distance threshold — items whose
fingerprints differ by at most that many bits are grouped together.

```go
sim := dedup.NewDetector[Doc](dedup.SimHash, func(x Doc) string { return x.Body })
sim.Index([]Doc{
	{ID: "a", Body: "the quick brown fox jumps over the lazy dog"},
	{ID: "b", Body: "the quick brown fox jumps over the lazy dog today"},
})

for _, group := range sim.FindNearDuplicates(5) {
	fmt.Printf("near-duplicate cluster of %d\n", len(group))
}
```

A threshold of `5` is a sensible default for short text. Lower it to demand
closer matches; raise it to cast a wider net.

## Hashing strategies

Pass one of these to `NewDetector` (or `NewContentHasher`):

| Strategy         | Matches when…                                       | Near-dup? |
| ---------------- | --------------------------------------------------- | :-------: |
| `SimpleHash`     | content is byte-for-byte identical                  |     —     |
| `NormalizedHash` | content is equal ignoring case and whitespace       |     —     |
| `SemanticHash`   | the meaningful lines match after structural tagging  |     —     |
| `SimHash`        | content is *similar* — within a Hamming threshold   |     ✓     |

Rule of thumb: start with `NormalizedHash` for "are these the same document?",
reach for `SimHash` when you need "are these roughly the same?".

`NormalizedHash` treats Unicode whitespace (including NBSP and EM SPACE) like
ASCII whitespace. `SimHash` retains Unicode letters, numbers, and combining
marks while collapsing Unicode whitespace and dropping ordinary punctuation.
Inputs shorter than the configured shingle size contribute their complete
normalized content as one feature; punctuation-only non-whitespace inputs use a
distinct raw-content fallback instead of collapsing to the empty fingerprint.
Persisted SimHash fingerprints for those previously defective cases must be
rebuilt after upgrading; long ASCII fingerprints are unchanged.

## Stats, metadata, and ordering

`Stats` returns a typed summary of the last index — total items, unique hashes,
duplicate groups, and a deduplication rate. `FindDuplicateGroups` returns the
same duplicate item groups as `FindDuplicates`, plus the shared hash for each
group. `WithOrdering` controls how items are sorted inside each group
(chainable):

```go
d := dedup.NewDetector[Doc](dedup.NormalizedHash, func(x Doc) string { return x.Body }).
	WithOrdering(func(a, b Doc) bool { return a.ID < b.ID })
d.Index(docs)

stats := d.Stats()
fmt.Printf("%d of %d items were duplicates\n", stats.DuplicateFiles, stats.TotalFiles)

for _, group := range d.FindDuplicateGroups() {
	fmt.Printf("%s has %d copies\n", group.Hash, len(group.Items))
}
```

`GetStats` remains available for callers that need the legacy `map[string]any`
shape.

## Choosing a canonical per group

`Canonicalize` collapses each duplicate group to a single survivor. Give it a
`better(a, b)` predicate and it keeps the preferred element per group (ties keep
the earlier one). `CanonicalizeWith` adds a merge step to fold losers into the
winner first.

```go
groups := d.FindDuplicates() // [][]Doc

// Keep the shortest body in each group.
keep := dedup.Canonicalize(groups, func(a, b Doc) bool {
	return len(a.Body) < len(b.Body)
})
fmt.Printf("%d unique survivors\n", len(keep))
```

## Lifecycle and concurrency

- Call order is `NewDetector` → optional `WithOrdering` → `Index` → query.
- `Index` **replaces** the index every call; it never appends.
- A `Detector` holds no locks. Index once on a single goroutine, then run as
  many concurrent reads (`FindDuplicates`, `FindNearDuplicates`, `GetStats`) as
  you like. Do not call `Index` while reads are in flight.

See **[docs/api-reference.md](docs/api-reference.md)** for the full API, edge
cases, and the precise behavior of each strategy.

## License

[MIT](LICENSE) © DotCommander contributors

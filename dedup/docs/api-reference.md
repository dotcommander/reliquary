# api reference

```go
import "github.com/dotcommander/reliquary/dedup"
```

Detect exact and near-duplicate items in a slice of any type using content-based hashing.

```go
type Doc struct{ ID, Body string }

docs := []Doc{
    {ID: "a", Body: "Hello world"},
    {ID: "b", Body: "Hello world"},
    {ID: "c", Body: "Entirely different"},
}

d := dedup.NewDetector[Doc](dedup.NormalizedHash, func(doc Doc) string { return doc.Body })
d.Index(docs)

for _, group := range d.FindDuplicates() {
    fmt.Printf("duplicates: %v\n", group) // [{a Hello world} {b Hello world}]
}
```

---

## lifecycle-and-call-order

`NewDetector` initializes an empty index. Call `Index` before querying.

```
NewDetector → [WithOrdering] → Index → FindDuplicates / FindDuplicateGroups / FindNearDuplicates / Stats / GetStats
```

**`Index` replaces, never appends.** Each call discards the previous index and rebuilds it from the provided slice. To add items incrementally, collect them into a single slice and call `Index` once.

**Querying before `Index`** is safe. `NewDetector` initializes `hashIndex` to an empty map, so calling `FindDuplicates`, `FindDuplicateGroups`, `FindNearDuplicates`, `Stats`, or `GetStats` on a freshly created detector returns empty results, not a panic.

**Calling `Index` again** replaces the prior index entirely. The previous items are discarded.

---

## thread-safety

`Detector` and `ContentHasher` contain no mutexes or atomics and are **not safe for concurrent use**.

- Do not call `Index` concurrently with any other method.
- Do not call `FindDuplicates`, `FindNearDuplicates`, or `GetStats` concurrently with `Index` or with each other when writes may occur.

**Safe pattern:** index on a single goroutine, then read from multiple goroutines only after `Index` returns.

```go
// safe: Index completes before any concurrent reads
d.Index(items)

var wg sync.WaitGroup
wg.Add(2)
go func() { defer wg.Done(); process(d.FindDuplicates()) }()
go func() { defer wg.Done(); report(d.GetStats()) }()
wg.Wait()
```

If you need concurrent indexing of independent datasets, create one `Detector` per goroutine.

---

## hashing-strategies

`HashingStrategy` is a `string` type. Pass one of the four constants to `NewDetector` or `NewContentHasher`.

| Constant | Value | Behavior |
|---|---|---|
| `SimpleHash` | `"simple"` | SHA-256 of the raw content string; byte-identical content only. |
| `NormalizedHash` | `"normalized"` | Lowercases the content and collapses all whitespace to single spaces before hashing. Two strings differing only in case or spacing are treated as duplicates. |
| `SemanticHash` | `"semantic"` | Parses the content line-by-line, tagging Markdown headers (`# …` → `HEADER:`), list items (`-`/`*` → `LIST:`), code fences (` ``` ` → `CODE_BLOCK`), and body text (` TEXT:`). Blank lines are dropped. Produces identical hashes for documents with the same structural elements in the same order, ignoring decorative whitespace. |
| `SimHash` | `"simhash"` | Produces a 64-bit locality-sensitive hash via character 3-gram (shingle) features. Similar content yields hashes with few differing hex characters. **Required** for `FindNearDuplicates`. |

---

## content-hasher

`ContentHasher` applies a single strategy to arbitrary strings. Use it directly when you need hashes outside of a detection pipeline.

### NewContentHasher

```go
func NewContentHasher(strategy HashingStrategy) *ContentHasher
```

Returns a `*ContentHasher` configured with `strategy`. The struct has no exported fields.

### HashContent

```go
func (ch *ContentHasher) HashContent(content string) string
```

Returns a 16-character lowercase hex string. For `SimHash` the string is a 16-hex-digit (64-bit) fingerprint; for all other strategies it is the first 16 hex characters of a SHA-256 digest.

```go
h := dedup.NewContentHasher(dedup.SimpleHash)
fmt.Println(h.HashContent("hello")) // e.g. "2cf24dba5fb0a30e"
```

---

## detector

`Detector[T any]` indexes a slice of items by content hash and exposes duplicate and near-duplicate queries. It holds no exported fields.

### Stats

```go
type Stats struct {
    TotalFiles        int
    UniqueHashes      int
    DuplicateGroups   int
    DuplicateFiles    int
    DeduplicationRate float64
}
```

`Stats` is a typed snapshot of the current index. Prefer this type when writing new code.

### DuplicateGroup

```go
type DuplicateGroup[T any] struct {
    Hash  string
    Items []T
}
```

`DuplicateGroup` contains the shared content hash and cloned duplicate items for one exact-match group.

### NewDetector

```go
func NewDetector[T any](strategy HashingStrategy, content func(T) string) *Detector[T]
```

- `strategy` — the hashing strategy to apply.
- `content` — extracts the string to hash from each item; called once per item during `Index`.

```go
d := dedup.NewDetector[string](dedup.SimpleHash, func(s string) string { return s })
```

### WithOrdering

```go
func (d *Detector[T]) WithOrdering(less func(a, b T) bool) *Detector[T]
```

Sets the comparator used to sort items within each duplicate group returned by `FindDuplicates`. Returns `d` for fluent chaining. If not called, items within a group appear in their original insertion order (stable).

```go
type File struct{ Name, Content string }

d := dedup.NewDetector[File](dedup.NormalizedHash, func(f File) string { return f.Content }).
    WithOrdering(func(a, b File) bool { return a.Name < b.Name })
```

### Index

```go
func (d *Detector[T]) Index(items []T)
```

Builds (or rebuilds) the internal hash index from `items`. Each call replaces the previous index entirely. Call `Index` before `FindDuplicates`, `FindNearDuplicates`, or `GetStats`.

### FindDuplicates

```go
func (d *Detector[T]) FindDuplicates() [][]T
```

Returns all groups of two or more items that share an identical hash. Each inner slice is one group of exact duplicates. Groups are ordered largest-first (most duplicates first). Within a group, items are sorted by the `WithOrdering` comparator if one was set. Returns `nil` when no duplicates exist.

Works with all four strategies; for `SimHash`, two items must hash to the exact same 64-bit fingerprint to appear together here — use `FindNearDuplicates` for fuzzy matching.

```go
d.Index(items)
for _, group := range d.FindDuplicates() {
    fmt.Printf("group of %d: %v\n", len(group), group)
}
```

### FindDuplicateGroups

```go
func (d *Detector[T]) FindDuplicateGroups() []DuplicateGroup[T]
```

Returns exact duplicate groups with the shared hash included. Ordering and matching behavior are the same as `FindDuplicates`. The returned item slices are cloned, so sorting or mutating them does not mutate the detector's stored buckets.

Use this when callers need metadata for reports, cleanup previews, or stable audit output:

```go
d.Index(items)
for _, group := range d.FindDuplicateGroups() {
    fmt.Printf("%s: %d items\n", group.Hash, len(group.Items))
}
```

### FindNearDuplicates

```go
func (d *Detector[T]) FindNearDuplicates(threshold int) [][]T
```

Returns groups of items whose SimHash fingerprints are within `threshold` of each other. The distance metric is the bit-level Hamming distance between two 16-character, 64-bit fingerprint strings (range 0–64).

**Requires `SimHash` strategy.** Returns an empty, non-nil slice (`[][]T{}`) immediately when called on a detector using any other strategy.

A `threshold` of `0` is equivalent to exact matching. A `threshold` of `3` or `4` catches typical near-duplicates (minor edits, paraphrases).

```go
d := dedup.NewDetector[string](dedup.SimHash, func(s string) string { return s })
d.Index(corpus)
groups := d.FindNearDuplicates(3) // groups with <=3 differing fingerprint bits
```

### Stats

```go
func (d *Detector[T]) Stats() Stats
```

Returns a typed snapshot of the current index. All fields are always set.

| Field | Type | Description |
|---|---|---|
| `TotalFiles` | `int` | Total number of items indexed. |
| `UniqueHashes` | `int` | Number of distinct hashes in the index. |
| `DuplicateGroups` | `int` | Number of hash buckets containing two or more items. |
| `DuplicateFiles` | `int` | Total count of items that belong to a duplicate group. |
| `DeduplicationRate` | `float64` | `DuplicateFiles / TotalFiles`; `0.0` when `TotalFiles` is zero. |

```go
stats := d.Stats()
fmt.Printf("%.1f%% duplicates\n", stats.DeduplicationRate*100)
```

### GetStats

```go
func (d *Detector[T]) GetStats() map[string]any
```

Returns a legacy map snapshot of the current index. New code should prefer `Stats`.

| Key | Type | Description |
|---|---|---|
| `"total_files"` | `int` | Total number of items indexed. |
| `"unique_hashes"` | `int` | Number of distinct hashes in the index. |
| `"duplicate_groups"` | `int` | Number of hash buckets containing two or more items. |
| `"duplicate_files"` | `int` | Total count of items that belong to a duplicate group. |
| `"deduplication_rate"` | `float64` | `duplicate_files / total_files`; `0.0` when `total_files` is zero. |

```go
stats := d.GetStats()
fmt.Printf("%.1f%% duplicates\n", stats["deduplication_rate"].(float64)*100)
```

---

## choosing-a-strategy

| Goal | Strategy |
|---|---|
| Exact byte match | `SimpleHash` |
| Ignore case and whitespace differences | `NormalizedHash` |
| Ignore formatting, focus on Markdown structure | `SemanticHash` |
| Detect near-duplicates (edits, paraphrases) | `SimHash` + `FindNearDuplicates` |

**`SimpleHash`** — use when content must be byte-for-byte identical. A trailing space or different line ending produces a different hash. Good for detecting verbatim copies where formatting is controlled (e.g., database records, generated output).

**`NormalizedHash`** — use when content may differ only in case or whitespace (extra spaces, mixed indentation, Windows vs Unix line endings). Two strings that are identical after `strings.ToLower` and whitespace collapse hash equally. Does not strip punctuation or Markdown syntax.

**`SemanticHash`** — use for Markdown documents where you want to treat structurally equivalent content as duplicates even when decorative details differ. A document with an extra blank line, or with `*item*` reformatted to `- item`, may produce the same hash. Non-empty body-text lines contribute a `TEXT:` element after lowercasing and whitespace normalization, including very short lines.

**`SimHash`** — use when you expect near-duplicates: lightly edited versions, paraphrases, or documents where a paragraph was added or removed. `SimHash` is the only strategy that enables `FindNearDuplicates`. `FindDuplicates` still works with `SimHash` but requires an exact 64-bit fingerprint match, which is unlikely for truly different content.

Only `SimHash` supports `FindNearDuplicates`. Calling `FindNearDuplicates` with any other strategy returns an empty non-nil slice immediately.

---

## edge-cases

**Empty input slice**

```go
d := dedup.NewDetector[string](dedup.SimpleHash, func(s string) string { return s })
d.Index([]string{})          // valid; index is empty
d.FindDuplicates()           // returns nil
d.FindNearDuplicates(3)      // returns [][]T{} (SimHash) or [][]T{} (other strategies)
d.GetStats()                 // total_files: 0, deduplication_rate: 0.0
```

`Index` with an empty slice is safe. The for loop does not execute and the index is empty.

**Single item**

```go
d.Index([]string{"only one"})
d.FindDuplicates() // returns nil — a group of one is not a duplicate group
```

A bucket must contain more than one item to be returned as a duplicate group. A single-item index always produces `nil` from `FindDuplicates`.

**Multiple items with the same hash**

All items sharing a hash are placed in a single group. The threshold for a group to appear in results is strictly more than one item. There is no upper bound; a group can contain any number of items.

**`deduplication_rate` when `total_files` is zero**

`GetStats` returns `0.0` for `deduplication_rate` when `total_files` is `0`.

```go
stats := d.GetStats()
rate := stats["deduplication_rate"].(float64)
if stats["total_files"].(int) == 0 {
    // no items were indexed
}
```

**Calling `Index` a second time**

The second call discards all previous data. Only the items from the most recent `Index` call are visible to subsequent queries.

---

## full-example

```go
package main

import (
    "fmt"

    "github.com/dotcommander/reliquary/dedup"
)

type Doc struct {
    ID   string
    Body string
}

func main() {
    docs := []Doc{
        {ID: "1", Body: "The quick brown fox"},
        {ID: "2", Body: "the quick brown fox"}, // differs only in case
        {ID: "3", Body: "The quick brown fox jumps"}, // near-duplicate
        {ID: "4", Body: "Completely unrelated content"},
    }

    // Exact duplicates under normalized comparison (case + whitespace insensitive)
    exact := dedup.NewDetector[Doc](dedup.NormalizedHash, func(d Doc) string { return d.Body }).
        WithOrdering(func(a, b Doc) bool { return a.ID < b.ID })
    exact.Index(docs)
    fmt.Println("exact groups:", exact.FindDuplicates()) // docs 1 and 2 grouped together

    // Near-duplicates using SimHash
    near := dedup.NewDetector[Doc](dedup.SimHash, func(d Doc) string { return d.Body })
    near.Index(docs)
    fmt.Println("near groups:", near.FindNearDuplicates(4))

    // Stats
    stats := near.GetStats()
    fmt.Printf("indexed %d items, %d unique hashes\n",
        stats["total_files"].(int),
        stats["unique_hashes"].(int),
    )
}
```

# api guide — github.com/dotcommander/reliquary/chunking

```go
import "github.com/dotcommander/reliquary/chunking"

chunker, err := chunking.NewChunker(chunking.SmartBoundary)
if err != nil {
    log.Fatal(err)
}
chunks := chunker.Chunk(text, 1200, 100)
for _, ch := range chunks {
    fmt.Printf("[%d] %d chars  hash:%s\n", ch.ID, ch.CharCount, ch.ContentHash)
}
```

---

## contents

1. [core interface](#core-interface)
2. [strategies](#strategies)
3. [chunk struct fields](#chunk-struct-fields)
4. [token-based chunking](#token-based-chunking)
5. [semantic chunking](#semantic-chunking)
6. [token budgets](#token-budgets)
7. [hard limits and filtering](#hard-limits-and-filtering)
8. [provenance and span helpers](#provenance-and-span-helpers)
9. [optimal chunker](#optimal-chunker)
10. [utilities](#utilities)

---

## core interface

```go
type Chunker interface {
    Chunk(text string, size int, overlap int) []Chunk
    Strategy() Strategy
}
```

`size` is the target chunk size in characters (runes). `overlap` is the number
of characters carried forward from the previous chunk. Both are hints — the
chunker may produce chunks larger or smaller depending on natural boundaries.

Construct any rule-based chunker with:

```go
chunker, err := chunking.NewChunker(strategy)
```

Returns `ErrUnknownStrategy` when `strategy` is not registered. The `Semantic`
constant is not wired here — use `NewSemanticChunker` instead.

---

## strategies

| Strategy constant     | What it does                                                                                              | Sets `Path`? | Sets `Metadata`?    |
|-----------------------|-----------------------------------------------------------------------------------------------------------|:------------:|:-------------------:|
| `SmartBoundary`       | Cascades: paragraph → sentence → word. Best general-purpose choice.                                       | No           | No                  |
| `SentenceBoundary`    | Splits at sentence boundaries only.                                                                       | No           | No                  |
| `WordBoundary`        | Splits at word boundaries only.                                                                           | No           | No                  |
| `ParagraphAware`      | Splits at blank-line paragraph boundaries.                                                                | No           | No                  |
| `MarkdownAware`       | Goldmark-parsed: splits by markdown block, preserves table headers in every chunk.                        | No           | Yes                 |
| `HeadingAware`        | Goldmark-parsed: groups content under headings; each chunk carries its heading breadcrumb.                | Yes          | Yes                 |
| `HardCut`             | Cuts at exactly `size` runes, no boundary awareness.                                                      | No           | No                  |
| `TokenBased`          | Splits at tiktoken token boundaries (default encoding `cl100k_base`). Use `NewTokenChunker` for others.  | No           | No                  |
| `Optimal`             | Targets a configurable character band (default 5 000–15 000). Ignores `overlap`. See [optimal chunker](#optimal-chunker). | No | No |
| `Semantic`            | Topic-boundary detection via embedding similarity. Requires `NewSemanticChunker`. **Not** wired in `NewChunker`. | No | No |

`Metadata` keys when set by `MarkdownAware`/`HeadingAware`:

| Key             | Values / notes                                    |
|-----------------|---------------------------------------------------|
| `"type"`        | `"heading"`, `"code"`, `"table"`, `"paragraph"`  |
| `"heading_level"` | `"1"` – `"6"`                                  |
| `"language"`    | fenced code language, e.g. `"go"` (code blocks)  |
| `"line_count"`  | number of lines in block                          |
| `"word_count"`  | word count of block text                          |

---

## chunk struct fields

```go
type Chunk struct {
    ID          int
    Text        string
    CharCount   int               // rune count of Text
    WordCount   int
    TokenCount  int               // 0 unless FillTokenCounts was called
    StartChar   int               // byte offset into original text
    EndChar     int               // byte offset into original text (exclusive)
    Path        []string          // heading breadcrumb; nil unless HeadingAware
    Metadata    map[string]string // block metadata; nil unless MarkdownAware/HeadingAware
    ContentHash string            // first 16 hex chars of SHA-256(Text); always set
}
```

**Span contract.** When `EndChar > StartChar`:

```go
originalText[ch.StartChar:ch.EndChar] == ch.Text  // guaranteed
```

When a post-processing step (overlap injection, hard-limit splitting, table
header duplication) cannot produce a contiguous match, both fields are cleared
to `0`. **Always check `EndChar > StartChar` before slicing the original text.**

---

## token-based chunking

Use `TokenBased` when token boundaries should drive the primary split:

```go
chunker, err := chunking.NewChunker(chunking.TokenBased) // cl100k_base
chunks := chunker.Chunk(text, 500, 50)
```

Use `NewTokenChunker` to specify a different tiktoken encoding:

```go
chunker, err := chunking.NewTokenChunker("o200k_base")
if err != nil {
    log.Fatal(err) // error on unrecognised encoding name
}
chunks := chunker.Chunk(text, 500, 50)
```

Empty string defaults to `cl100k_base`. Encoders are cached by name after
first construction.

Populate `Chunk.TokenCount` on any slice with:

```go
if err := chunking.FillTokenCounts(chunks, "cl100k_base"); err != nil {
    log.Fatal(err)
}
```

Count tokens for a single string:

```go
n, err := chunking.CountTokens("hello world", "cl100k_base")
```

---

## semantic chunking

`SemanticChunker` detects topic boundaries by comparing embedding similarity
between consecutive sentences (or structural units). It does not implement
`Chunker`; call `ChunkSemantic` directly.

### wiring a batch embedder

Any type that satisfies `BatchEmbedder` works:

```go
type BatchEmbedder interface {
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}
```

Any application-owned embedding client can satisfy this narrow interface.

```go
// embedder satisfies chunking.BatchEmbedder
sc, err := chunking.NewSemanticChunker(embedder, chunking.SemanticOpts{})
if err != nil {
    log.Fatal(err) // ErrNilEmbedder if embedder is nil or holds a typed nil
}

chunks := sc.ChunkSemantic(ctx, text, 1200, 100)
//                                       ^^^^  ^^^
//                              fallbackSize   fallbackOverlap
//         used when text is too short for semantic analysis,
//         or when embedding fails — falls back to SmartBoundary
```

### SemanticOpts fields

| Field              | Type      | Default | Effect                                                                           |
|--------------------|-----------|---------|----------------------------------------------------------------------------------|
| `MaxChunkChars`    | `int`     | 1600    | Hard ceiling in runes per chunk. Oversized groups are merged into the next.      |
| `MinChunkChars`    | `int`     | 200     | Groups smaller than this are merged with neighbours before finalising.           |
| `BreakSensitivity` | `float64` | 1.0     | Stddev multiplier for break threshold. Higher → fewer breaks → larger chunks.    |
| `SmoothingWindow`  | `int`     | 0       | Centered moving-average window over similarity scores. 0 = disabled. Use odd values (e.g. 3). |
| `CoherenceWindow`  | `int`     | 0       | Require N coherent neighbours on each side of a candidate break. 0 = disabled. Recommended: 2. |

Zero value is valid; defaults are applied automatically.

```go
opts := chunking.SemanticOpts{
    MaxChunkChars:    2000,
    BreakSensitivity: 1.2,  // slightly fewer breaks
    SmoothingWindow:  3,
    CoherenceWindow:  2,
}
sc, err := chunking.NewSemanticChunker(embedder, opts)
```

Structural markers in text (headings, horizontal rules, conversation turn
prefixes, paragraph blocks) are used as semantic atoms when present, reducing
embedding API calls and preserving source byte spans where possible.

### planning with precomputed embeddings

Use `SemanticUnits` and `PlanSemanticChunks` when a caller already owns the
embedding pass and wants to reuse those vectors for semantic boundaries.

```go
units := chunking.SemanticUnits(text)
texts := make([]string, len(units))
for i, unit := range units {
    texts[i] = unit.Text
}

embeddings, err := embedder.EmbedBatch(ctx, texts)
if err != nil {
    log.Fatal(err)
}

plan, ok := chunking.PlanSemanticChunks(text, units, embeddings, chunking.SemanticPlanOptions{
    MaxChunkChars:    2000,
    MinChunkChars:    200,
    BreakSensitivity: 1.0,
    FallbackSize:     1200,
    FallbackOverlap:  100,
})
if !ok {
    // Use your fallback chunker; the embeddings were not suitable for planning.
}
chunks := plan.Chunks
```

`SemanticPlan` includes the accepted `Units`, cosine similarities between
adjacent embeddings, chosen `Breaks`, and final `Chunks`. The planner reads but
does not normalize or otherwise mutate the caller-owned embeddings. It rejects
too few units, embedding count mismatches, dimension mismatches, empty vectors,
zero vectors, and vectors containing NaN or infinity. It does not call an
embedder.

---

## token budgets

Two patterns cover different needs.

### pattern 1 — token-aware splitting as primary strategy

`TokenBased` chunker + `EnforceTokenLimits` for a post-pass that drops or
re-splits any chunk the counter rejects.

### pattern 2 — boundary strategy with token ceiling

Let a boundary strategy produce natural chunks, then enforce a token budget.
This is the recommended pattern for markdown and heading-aware content.

```go
base, err := chunking.NewChunker(chunking.MarkdownAware)
if err != nil {
    log.Fatal(err)
}

counter, err := chunking.NewTiktokenCounter("cl100k_base", 500) // 500 token limit
if err != nil {
    log.Fatal(err)
}

// ChunkWithTokenLimit = base.Chunk + EnforceTokenLimits in one call
chunks := chunking.ChunkWithTokenLimit(base, markdownText, 1600, 100, counter)
```

`NewTiktokenCounter("", 500)` uses the default `cl100k_base` encoding.

Count tokens manually via the counter:

```go
n := counter.CountTokens("some text")
```

### TokenCounter interface

`EnforceTokenLimits` and `ChunkWithTokenLimit` accept any `TokenCounter`:

```go
type TokenCounter interface {
    CountTokens(text string) int
}
```

`*TiktokenCounter` satisfies this interface. Supply your own implementation to
use a different tokeniser.

### span behaviour under token splitting

Pass-through chunks (within budget) retain their `StartChar`/`EndChar` spans.
Chunks that are re-split to fit the budget have their spans cleared to `0`.
Apply the `EndChar > StartChar` guard before slicing.

---

## hard limits and filtering

### EnforceHardLimits

Ensures no chunk exceeds a character ceiling. Oversized chunks are re-split
using cascading boundaries: paragraph → sentence → word → hard cut. Chunk IDs
are rebuilt sequentially; empty chunks are dropped; text order is preserved.

```go
opts := chunking.LimitOptions{
    MaxChars:     800,
    Overlap:      50,
    OriginalText: originalText, // set to propagate sub-spans back to source offsets
}
chunks = chunking.EnforceHardLimits(chunks, opts)
```

When `OriginalText` is set and the incoming chunk has a valid span, the
function attempts to locate each sub-chunk within the original byte range and
populate sub-spans. When it cannot, spans are cleared.

When `MaxChars <= 0`, the function returns the input unchanged.

### FilterProse

Drops code blocks and tables; drops short paragraphs (`< ProseWordFloor = 5`
words) and short headings (`< HeadingWordFloor = 3` words). Reads
`Chunk.Metadata["type"]` and `Chunk.Metadata["word_count"]`. Chunks without
`Metadata` are passed through unchanged.

```go
prose := chunking.FilterProse(chunks) // useful after MarkdownAware or HeadingAware
```

Typical pipeline for prose retrieval from markdown:

```go
chunker, _ := chunking.NewChunker(chunking.HeadingAware)
raw := chunker.Chunk(md, 1600, 100)
prose := chunking.FilterProse(raw)
```

---

## provenance and span helpers

These functions map chunks back to exact positions in the original source, even
when `Chunk.StartChar`/`EndChar` were cleared.

### ChunkSpan

```go
type ChunkSpan struct {
    Start int // byte offset (inclusive)
    End   int // byte offset (exclusive)
}
```

`content[span.Start:span.End]` is always the chunk text when the span is valid.

### ResolveChunkSpan

Resolves the byte span of a chunk within `content`. First validates the
embedded `StartChar`/`EndChar` fields; if those fail (cleared or stale), falls
back to `Locate` starting from `cursor`.

```go
cursor := 0
for _, ch := range chunks {
    span, ok := chunking.ResolveChunkSpan(originalText, ch, cursor)
    if !ok {
        // chunk text could not be located — skip or handle
        continue
    }
    startLine, endLine := chunking.LineRangeForSpan(originalText, span)
    fmt.Printf("chunk %d: lines %d–%d\n", ch.ID, startLine, endLine)
    cursor = chunking.NextChunkCursor(span)
}
```

### NextChunkCursor

Returns `span.Start + 1` — advances just past the beginning of the matched
span so overlapping chunks and repeated phrases can still be found without
re-matching the same position.

### Locate

Lower-level function used internally by `ResolveChunkSpan`. Finds `fragment`
within `content` starting at `cursor`.

```go
start, end, ok := chunking.Locate(content, fragment, cursor)
```

### LineForOffset

Returns the 1-based line number for a byte offset.

```go
line := chunking.LineForOffset(content, offset)
```

### LineRangeForSpan

Returns the inclusive 1-based line range for a span. A span ending exactly at
a newline reports the content line before that newline as the end line.

```go
startLine, endLine := chunking.LineRangeForSpan(content, span)
```

### full provenance example

```go
chunker, _ := chunking.NewChunker(chunking.HeadingAware)
chunks := chunker.Chunk(source, 1600, 0)

cursor := 0
for _, ch := range chunks {
    span, ok := chunking.ResolveChunkSpan(source, ch, cursor)
    if !ok {
        cursor = 0 // reset on miss; next Locate searches from beginning
        continue
    }
    startLine, endLine := chunking.LineRangeForSpan(source, span)
    fmt.Printf("chunk %d  lines %d–%d  heading: %v\n",
        ch.ID, startLine, endLine, ch.Path)
    cursor = chunking.NextChunkCursor(span)
}
```

---

## optimal chunker

`OptimalChunker` targets a character band rather than a fixed size, useful
when downstream models have wide context windows and you want to maximise
information density without splitting at arbitrary boundaries.

```go
oc := chunking.NewOptimalChunker()
// defaults: OptimalLength=10000, MinLength=5000, MaxLength=15000
chunks := oc.Chunk(text, 0, 0) // size and overlap args are ignored
```

`OptimalChunker` does not implement the `Chunker` interface — it exposes
`Chunk(text string, size int, overlap int) []Chunk` with the same signature
but the `size` and `overlap` arguments are silently ignored; the band
configured at construction time governs output.

### StrategicSample

Extracts a representative sample of `optimalLen` characters for preflight
inspection or quality checks:

```go
sample := chunking.StrategicSample(text, 10000)
```

---

## utilities

### SplitSentences

Splits text into sentences using the same boundary logic the sentence-aware
chunkers use internally.

```go
sentences := chunking.SplitSentences(text)
```

### errors

| Error                  | When                                                        |
|------------------------|-------------------------------------------------------------|
| `ErrUnknownStrategy`   | `NewChunker` receives an unregistered strategy string.      |
| `ErrNilEmbedder`       | `NewSemanticChunker` receives a nil or typed-nil embedder.  |

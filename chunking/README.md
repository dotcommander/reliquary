# chunking

Text chunking utilities for Go.

This module provides deterministic chunking strategies for retrieval, embedding
pipelines, summarization, and LLM context assembly.

For the complete API reference — every strategy, the span contract, semantic options, and token budgets — see [docs/api-guide.md](docs/api-guide.md).

## Install

```sh
go get github.com/dotcommander/reliquary/chunking
```

## Usage

```go
package main

import (
	"fmt"
	"log"

	"github.com/dotcommander/reliquary/chunking"
)

func main() {
	chunker, err := chunking.NewChunker(chunking.SmartBoundary)
	if err != nil {
		log.Fatal(err)
	}

	chunks := chunker.Chunk("First paragraph. Second paragraph.", 1200, 100)
	fmt.Println(len(chunks))
}
```

## Strategies

- `SmartBoundary`
- `SentenceBoundary`
- `WordBoundary`
- `MarkdownAware`
- `HeadingAware`
- `ParagraphAware`
- `HardCut`
- `TokenBased`
- `Semantic`

`Semantic` chunking is created with `NewSemanticChunker` because it needs an
embedder that satisfies:

```go
type BatchEmbedder interface {
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}
```

## Source Spans

Each `Chunk` has `StartChar` and `EndChar` fields that are byte offsets into
the original input text. When set, the chunk text appears verbatim in the
source:

```go
text[ch.StartChar:ch.EndChar] == ch.Text
```

Spans are cleared (set to `0, 0`) when a post-processing step such as overlap
or hard-limit splitting cannot map the chunk text back to an exact contiguous
source slice. Always check for non-zero spans before using them.

## Separator Profiles

`DefaultTextSeparatorProfile` and `CJKThaiSeparatorProfile` expose ordered,
pure separator lists for callers that run recursive splitting outside this
package. The CJK/Thai profile includes ideographic punctuation, fullwidth
punctuation, and zero-width space so scripts without word-boundary spaces can
avoid poor fallback splits.

## Token Chunking

```go
// Default encoding (cl100k_base)
chunker, _ := chunking.NewChunker(chunking.TokenBased)
chunks := chunker.Chunk(text, 500, 50)
```

To use a different tiktoken encoding:

```go
// Specific encoding
chunker, err := chunking.NewTokenChunker("o200k_base")
if err != nil {
	log.Fatal(err)
}
chunks := chunker.Chunk(text, 500, 50)
```

`NewTokenChunker` validates the encoding at construction time and returns an
error for unrecognized names. Empty encoding defaults to `cl100k_base`.

## Token Counting

```go
tokens, err := chunking.CountTokens("hello world", "cl100k_base")
```

Token encoders are cached by encoding name.

## Token Budget Composition

Use `TokenBased` when token boundaries should drive the primary split.
Use `ChunkWithTokenLimit` when a normal boundary strategy (MarkdownAware,
HeadingAware, SmartBoundary, etc.) should drive the split, but final chunks
must fit a model-specific token budget.

```go
base, err := chunking.NewChunker(chunking.MarkdownAware)
if err != nil {
	log.Fatal(err)
}

counter, err := chunking.NewTiktokenCounter("cl100k_base", 500)
if err != nil {
	log.Fatal(err)
}

chunks := chunking.ChunkWithTokenLimit(base, markdownText, 1600, 100, counter)
```

Source spans keep the existing contract: pass-through chunks preserve spans;
token-split chunks may have unknown spans (zero). Construct with
`NewTiktokenCounter("", 500)` to use the default encoding (`cl100k_base`).

## Markdown Table Context

`MarkdownAware` splits oversized markdown tables by rows while preserving
the header row in every chunk. This retains column context for embedding and
retrieval. Table chunks use zero spans since the header is duplicated in
later chunks.

```go
chunker, _ := chunking.NewChunker(chunking.MarkdownAware)
chunks := chunker.Chunk(tableMarkdown, 200, 0)
// Each chunk contains the header row followed by body rows.
```

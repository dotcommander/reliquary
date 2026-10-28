# Fixed ingestion chunking policies

`WithChunking` configures chunkservice orchestration for an App's lifetime.
Without it, `WithChunker` and the existing ingestion defaults retain their behavior.
The two options cannot be combined, in either order. Repeated `WithChunking`
replaces the prior policy.

Supply a positive `ChunkSize` and `0 <= Overlap < ChunkSize`. An empty strategy
selects `smart_boundary`. Size and overlap retain their strategy-specific units;
`MaxTokens` is a separate ceiling on the final text sent for retrieval embedding
and indexed. Zero disables that ceiling. When tokenization is needed, an empty
encoding selects `cl100k_base`.

## Semantic boundaries

Use your production embedder and a caller-chosen index identity:

```go
app, err := reliquary.New(
    reliquary.WithEmbedder(embedder),
    reliquary.WithIndexIdentity("my-model:semantic-policy-v1"),
    reliquary.WithChunking(reliquary.ChunkingOptions{
        Strategy: chunking.Semantic,
        ChunkSize: 512,
        Overlap: 32,
        Semantic: &chunkservice.SemanticOptions{},
    }),
)
```

Semantic boundary embeddings use the App's embedding model and document input
kind. A boundary embedding failure is an ingestion error; there is no silent
boundary-based fallback. Final chunk text is embedded separately through the
usual retrieval path. All documents finish chunking and final embedding before
one atomic replacement, so a later document's failure leaves earlier revisions
intact. Empty normalized text still removes the previous document revision.

## Multilingual recursive splitting with a hard ceiling

```go
app, err := reliquary.New(
    reliquary.WithEmbedder(embedder),
    reliquary.WithIndexIdentity("my-model:cjk-thai-512-cap128-v1"),
    reliquary.WithChunking(reliquary.ChunkingOptions{
        Strategy: chunking.Recursive,
        ChunkSize: 512,
        Overlap: 32,
        SeparatorProfile: chunkservice.ProfileCJKThai,
        TokenEncoding: chunkservice.EncodingCL100K,
        MaxTokens: 128,
    }),
)
```

Named profiles are `default_text` and `cjk_thai`. Alternatively, set
`CustomSeparators: []string{"\n\n", "。", "\n", " "}`. Custom separators retain
that order; a rune fallback (`""`) is appended if omitted and may only appear
last. A nil slice means absent; an explicitly empty slice is invalid. Named and
custom profiles are mutually exclusive and require the recursive strategy.
Language is never detected automatically.

Optional `PreflightCounter` processing runs before the tokenizer-backed ceiling,
with chunkservice's existing strategy-specific behavior. It is neither provider
billing usage nor a total ingestion cost limit. The counter remains caller-owned
and must be behaviorally stable throughout the App's lifetime and support the
caller's concurrency pattern. Semantic options and separator slices are copied
when options are applied.

A final-chunk ceiling does **not** promise that an assembled prompt fits a model
context window. Budget for instructions, headers, citations, multiple retrieved
chunks, and output allowance separately.

## Index identity and policy changes

Your index identity must describe the effective embedding and chunking policy:
model, strategy, size/overlap, semantic options, encoding/ceiling, ordered
separators, and any custom counter's policy/version. Reliquary never derives or
rewrites this identity. A changed policy requires a new identity and either a
fresh index or an explicit reset and reingestion. Existing index identity mismatch
rejection remains in force, even for embeddings with matching dimensions.

There are no per-ingest policy overrides or injectable chunkservice instances.

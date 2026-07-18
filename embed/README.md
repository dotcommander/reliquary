# embed

```go
embedder := embed.NewHashing(64)
result, err := embedder.Embed(ctx, embeddings.Request{Inputs: []string{"hello"}})
```

`embed` provides a deterministic hashing embedder for examples, tests, and
`reliquary.Quickstart`.

It has no model runtime, API key, provider SDK, network dependency, or
production quality claim.

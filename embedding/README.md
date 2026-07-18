# embeddings

```go
model := embeddings.ModelRef{Provider: "local", Name: "demo", Dim: 3}
key := embeddings.CacheKey(model, "hello")
err := embeddings.ValidateDimensions([]embeddings.Vector{{1, 2, 3}}, model.Dim)
```

`embeddings` is the provider-neutral contract for embedding requests and dense
vectors.

The package does not call models, choose providers, or own vector-space policy.
Callers pass an `Embedder` implementation and keep model identity, dimensions,
fallback behavior, and cache invalidation explicit.

# embeddings

```go
model := embeddings.ModelRef{Provider: "local", Name: "demo", Dim: 3}
key := embeddings.CacheKey(model, "hello")
err := embeddings.ValidateDimensions([]embeddings.Vector{{1, 2, 3}}, model.Dim)

request := embeddings.Request{Model: model, Inputs: []string{"hello"}}
result := embeddings.Result{Model: model, Vectors: []embeddings.Vector{{1, 2, 3}}}
err = embeddings.ValidateResult(request, result)
```

`embeddings` is the provider-neutral contract for embedding requests and dense
vectors.

The package does not call models, choose providers, or own vector-space policy.
Callers pass an `Embedder` implementation and keep model identity, dimensions,
fallback behavior, and cache invalidation explicit.

A successful `Embed` call returns exactly one positive-dimension, finite vector
per input in the same order. `ValidateResult` checks that cardinality and shape,
including agreement with declared request and result dimensions; zero-magnitude
vectors remain valid. Implementations should run the reusable
`embedding/embeddingtest` contract suite.

`ModelRef.Identity` uses versioned byte-length framing. The `modelref:v1`
format prevents delimiter collisions, including with Unicode fields. Upgrading
to this version intentionally invalidates cache keys produced by the legacy
colon-joined identity; callers should expect cold misses rather than dual-read
old keys.

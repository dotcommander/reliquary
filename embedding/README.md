# embedding

```go
model := embedding.ModelRef{Provider: "local", Name: "demo", Dim: 3}
key := embedding.CacheKey(model, "hello")
err := embedding.ValidateDimensions([]embedding.Vector{{1, 2, 3}}, model.Dim)

request := embedding.Request{Model: model, Inputs: []string{"hello"}}
result := embedding.Result{Model: model, Vectors: []embedding.Vector{{1, 2, 3}}}
err = embedding.ValidateResult(request, result)
```

`embedding` is the provider-neutral contract for embedding requests and dense
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

## Cache decorator

`embedding/cache` wraps any `Embedder` and caches one vector per exact input.
You inject the Store and a nonblank identity:

```go
store := embeddingcacheinmem.New()
cached, err := embeddingcache.New(base, store, embeddingcache.Config{
	Identity: "openai:text-embedding-3-small:1536:input-v1",
})
```

The decorator performs one lookup per unique input, embeds unique misses in one
batch, restores original order, and clones vectors across Store, embedder, and
caller ownership boundaries. It validates hits, misses, and the assembled
result before writing. Store and embedder errors fail closed.

The identity is caller-owned invalidation policy. Change it whenever provider,
model, revision, dimensions, preprocessing, or output semantics change. The
decorator derives opaque keys from that identity, the requested `ModelRef`, and
the exact input bytes. The decorator itself provides no persistence, TTL,
eviction, singleflight, or background work.

For process-local caching, `embedding/cache/inmem` provides a concurrency-safe
Store with an immediately usable zero value. It clones vectors on reads and
writes and intentionally provides no persistence, capacity, TTL, eviction,
statistics, or lifecycle methods.

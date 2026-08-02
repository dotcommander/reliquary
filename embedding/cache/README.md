# embedding/cache

`embedding/cache` adds caller-owned, provider-neutral caching to an
`embedding.Embedder`. Give it a cache identity that changes whenever embedding
output compatibility changes, and provide a Store implementation suitable for
your process and concurrency needs.

```go
store := inmem.New()
cached, err := embeddingcache.New(base, store, embeddingcache.Config{
	Identity: "openai:text-embedding-3-small:1536:input-v1",
})
```

Entries are keyed by the cache identity, requested model, and exact input
bytes. The decorator preserves request order, groups duplicate inputs, validates
cached and generated vectors, and clones vectors at ownership boundaries. It
does not provide persistence, eviction, invalidation, or singleflight.

`embedding/cache/inmem` supplies an optional concurrency-safe Store for
process-local caches. Its zero value works, and `inmem.New()` constructs an
initialized Store. Use a different Store when entries must survive the process
or need capacity, TTL, or eviction policy.

# embedding/cache/inmem

```go
store := inmem.New()
cached, err := embeddingcache.New(base, store, embeddingcache.Config{
	Identity: "openai:text-embedding-3-small:1536:input-v1",
})
```

`embedding/cache/inmem` is a concurrency-safe, process-local implementation of
`cache.Store`. It clones vectors on writes and reads so callers, the decorator,
and the Store do not share mutable backing arrays. The zero `Store` value is
ready to use.

Entries live only for the lifetime of the Store. The package provides no TTL,
eviction, capacity limit, persistence, statistics, or lifecycle methods.

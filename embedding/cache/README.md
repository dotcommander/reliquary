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
does not provide persistence, eviction, or invalidation.

`embedding/cache/inmem` supplies an optional concurrency-safe Store for
process-local caches. Its zero value works, and `inmem.New()` constructs an
initialized Store. Use a different Store when entries must survive the process
or need capacity, TTL, or eviction policy.

Set `CoalesceRequests: true` in `Config` to share identical concurrent nonempty
requests within this decorator. The default is false. Exact ordered inputs,
request kind, model and cache identity define a flight; overlapping batches
remain independent and provider batching is preserved.

The first caller runs synchronously with its own context. Followers can cancel
without stopping it; leader cancellation can fail followers. Failures are shared
only with current followers; later calls retry. If the leader panics or calls
`runtime.Goexit`, followers receive `ErrCoalescedCallAborted` while the leader
retains that behavior. Returned vectors are independent for every caller.

Use one decorator only for callers with interchangeable context-dependent provider
and store semantics. Separate tenant or authentication boundaries need separate
decorators. Stores and providers still own synchronization for distinct requests.

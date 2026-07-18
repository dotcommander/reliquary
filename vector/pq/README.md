# vectors/pq

```go
quantizer, err := pq.NewQuantizer(pq.DefaultConfig())
```

`vectors/pq` implements product quantization primitives for dense vectors.

It is a vector-space mechanism only. Callers own embedding model identity,
training corpus selection, index manifests, recall targets, and invalidation
when vectors or codebooks change.

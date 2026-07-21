# vectors/pq

```go
quantizer, err := pq.NewQuantizer(pq.DefaultConfig())
```

`vectors/pq` implements product quantization primitives for dense vectors.

It is a vector-space mechanism only. Callers own embedding model identity,
training corpus selection, index manifests, recall targets, and invalidation
when vectors or codebooks change.

Training rejects ragged and non-finite vectors. Decode and asymmetric-distance
operations reject out-of-range codes; `DistanceWithTables` retains its
documented panic contract for invalid precomputed inputs. Serialized codebooks
are limited to 256 MiB and 65,536 subspaces, which also bounds allocation count
for hostile headers. `Load` rejects malformed flags, dimensions, truncated or
non-finite codebooks, and leaves the receiving quantizer unchanged on every
failure.

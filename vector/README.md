# vectors

Zero-dependency primitive vector math for embedding and retrieval systems in Go.

Cosine similarity, normalization, binary quantization, in-memory search indexes,
pooling, rank fusion, and clustering — `float32` for hot embedding paths,
`float64` where precision governs. Standard library only.

```bash
go get github.com/dotcommander/reliquary/vector
```

```go
package main

import (
	"fmt"

	"github.com/dotcommander/reliquary/vector"
)

func main() {
	a := []float32{0.10, 0.20, 0.30}
	b := []float32{0.20, 0.10, 0.40}

	fmt.Println(vectors.Cosine32(a, b)) // cosine similarity in [-1, 1]
}
```

## Why

Embedding pipelines need the same handful of operations over and over: compare
two vectors, normalize a batch, shrink vectors to bits for fast candidate
generation, fuse rankings from multiple retrievers. `vectors` provides those as
small, allocation-conscious functions with no dependency footprint — drop it
into any module without dragging in a math framework.

## What's inside

| Area | Highlights |
| --- | --- |
| **Similarity & distance** | `Cosine32/64`, `CosineDistance32/64`, `Dot32/64`, `Euclidean64`, `JaccardWords` |
| **Normalization** | in-place `Normalize32/64`, non-mutating `NormalizeTo32/64`, `IsUnit32/64` |
| **Binary quantization** | `Quantize`, `ComputeMedians`, `HammingDistance`, `BinaryVector` |
| **Blob storage** | `EncodeFloat32Vec` / `DecodeFloat32Vec`, `DotFromBlob` and `TopKFromBlob` for on-disk vectors |
| **In-memory search** | `ExactIndex` (exact cosine), `SearchKeys` rerank, `BinaryIndex` (Hamming candidate generation), checked build reports |
| **Pooling** | `MeanPool32`, `WeightedMeanPool32` |
| **Rank fusion** | `RRF` — reciprocal rank fusion across multiple ranked lists |
| **Clustering** | root `KMeans` (`float32`) and the [`clustering`](docs/clustering.md) subpackage (k-means, HAC, silhouette) |
| **Semantic boundaries** | sliding-window similarity, elbow/curvature detection, adaptive thresholds |
| **Artifact identity** | index manifests, ANN profile result records, and transform/profile digests |

### Binary quantization at a glance

```go
medians := vectors.ComputeMedians(corpus)        // per-dimension thresholds
q, _ := vectors.Quantize(query, medians)         // []float32 -> BinaryVector
hamming := vectors.HammingDistance(q, candidate) // fast, branch-light
```

### Near-duplicate detection

Group embeddings that are close in cosine space — the **semantic** complement to
the lexical SimHash near-dup primitive in the `dedup` module.

```go
groups := vectors.NearDuplicateGroups(corpus, 0.92) // [][]int of mutual near-dups
for _, g := range groups {
	fmt.Println("near-duplicate cluster:", g) // indices into corpus, size >= 2
}

pairs := vectors.NearDuplicatePairs(corpus, 0.92) // [][2]int linked (i<j) pairs
```

`NearDuplicateGroups` returns connected components (singletons omitted);
`NearDuplicatePairs` returns the raw linked pairs. v1 verifies every pair with
exact `Cosine32` (a brute-force-correct baseline), so no true positive is
dropped; binary quantization is wired in as the O(n) screen for a future
radius-bounded prefilter.

### Index and ANN profile identity

`IndexManifest` records caller-owned embedding, chunking, and profile identity.
`HashIndexProfile` remains the legacy stable hex helper for persisted profile
hashes. New callers can use `HashIndexProfileIdentity(fields).String()` for an
ordered transform digest and store it in `IndexProfileHash`.

`ANNProfile` and `ANNProfileResult` describe candidate limits, oversampling,
quantization labels, exact-rescore flags, memory estimates, recall@K, and
latency summaries without importing an ANN engine or vector database.

### Reciprocal rank fusion

```go
// Fuse rankings from two retrievers (e.g. dense + lexical).
fused, _ := vectors.RRF([][]int{denseRanks, lexicalRanks}, 60)
for _, s := range fused {
	fmt.Println(s.Index, s.Score)
}
```

## `float32` vs `float64`

Use `float32` for embedding and retrieval — it halves memory and is faster on
modern hardware. Reach for `float64` only when numerical precision is the
governing constraint (e.g. iterative clustering math). The two spaces are not
interchangeable; pick one per pipeline.

## Documentation

- [`docs/api-reference.md`](docs/api-reference.md) — every exported symbol in the root package, with semantics and edge cases.
- [`docs/clustering.md`](docs/clustering.md) — the `clustering` subpackage (`ClusterService`, k-means, HAC, silhouette analysis).
- `go doc github.com/dotcommander/reliquary/vector` — package overview from source.

## License

[MIT](LICENSE) © DotCommander contributors

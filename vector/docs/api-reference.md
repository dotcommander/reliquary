# api reference — `github.com/dotcommander/reliquary/vector`

```go
import "github.com/dotcommander/reliquary/vector"

a := []float32{0.6, 0.8}          // already unit-length
b := []float32{0.8, 0.6}

sim := vectors.Cosine32(a, b)     // 0.96
dist := vectors.CosineDistance32(a, b) // 0.04

vectors.Normalize32(a)             // mutates a in place
norm := vectors.NormalizeTo32(b)   // returns a new slice, b unchanged
```

All cosine and index APIs assume **L2-normalized inputs**. Passing unnormalized vectors produces mathematically valid but meaningless similarity values.

---

## similarity & distance

### `Cosine32(a, b []float32) float32`

Cosine similarity between two float32 vectors. Accumulates in float64 to reduce rounding error on high-dimensional inputs. Returns `0` on length mismatch or zero-magnitude input.

### `Cosine64(a, b []float64) float64`

float64 twin of `Cosine32`. Uses scaled accumulation to avoid overflow and underflow for finite inputs. Returns `0` on length mismatch or zero-magnitude input.

### `CosineDistance32(a, b []float32) float32`

`1 - cosine(a, b)`. Returns `1` on length mismatch, empty input, zero-magnitude, NaN, or Inf inputs.

### `CosineDistance64(a, b []float64) float64`

float64 twin of `CosineDistance32`. Same edge-case returns.

### `Dot32(a, b []float32) float64`

Dot product of two float32 vectors, accumulated in float64. For L2-normalized vectors the result equals cosine similarity. Returns `0` on length mismatch.

### `Dot64(a, b []float64) float64`

Dot product of two float64 vectors. Returns `0` on length mismatch.

### `Euclidean64(a, b []float64) float64`

L2 distance between two float64 vectors. Returns `math.Inf(1)` if either vector is empty or lengths differ.

---

## normalization

### `Normalize32(v []float32) float32`

L2-normalizes `v` in place using scaled accumulation so extreme finite values remain normalizable. Returns the original magnitude, which may be `+Inf` when the mathematical magnitude exceeds the float32 range. Zero-magnitude input is left unchanged and returns `0`.

### `Normalize64(v []float64) float64`

float64 twin of `Normalize32`. Its returned magnitude may similarly be `+Inf` when the mathematical result exceeds the float64 range.

### `NormalizeTo32(v []float32) []float32`

Returns a new L2-normalized copy of `v` without mutating the input. Uses scaled accumulation so extreme finite values remain normalizable. If `v` has zero magnitude, returns the input slice unchanged (identity preserved, not a fresh zero slice).

### `NormalizeTo64(v []float64) []float64`

float64 twin of `NormalizeTo32`. Same zero-magnitude behavior.

### `NormSquared32(v []float32) float64`

Squared L2 norm of `v`, accumulated in float64 to avoid intermediate rounding error.

### `NormSquared64(v []float64) float64`

Squared L2 norm of a float64 vector.

### `IsUnit32(v []float32, tolerance float64) bool`

Reports whether `v` is a unit vector within the given tolerance, i.e. `|‖v‖² − 1| ≤ tolerance`. Returns `false` for empty vectors, zero vectors, NaN/Inf inputs, negative tolerances, and non-finite tolerances.

### `IsUnit64(v []float64, tolerance float64) bool`

float64 twin of `IsUnit32`. Same validation rules.

### `ComputeCentroid64(points [][]float64) []float64`

Returns the mean vector of `points`. Returns `nil` if `points` is empty.

---

## quantization & binary

Binary quantization encodes a float32 vector into a `BinaryVector` (packed `[]uint64`) using per-dimension thresholds. Hamming distance on binary vectors approximates cosine distance at ~64× lower memory cost.

```go
corpus := [][]float32{embedA, embedB, embedC}   // L2-normalized
thresholds := vectors.ComputeMedians(corpus)     // per-dimension medians

bv, err := vectors.Quantize(embedA, thresholds)
if err != nil {
    // dimension mismatch — embedder changed since thresholds were built
}

dist := vectors.HammingDistance(bv, bvOther)     // lower = more similar
```

### `type BinaryVector []uint64`

Packed bit representation of a float32 vector. A 768-dimension embedding occupies 12 `uint64`s; 1024 dimensions occupies 16. Bit `i` is stored at `words[i/64] & (1 << (i%64))`.

### `BinaryWords(dim int) int`

Number of `uint64` words required to represent `dim` bits. Returns `0` for `dim ≤ 0`.

### `Quantize(vec, thresholds []float32) (BinaryVector, error)`

Allocates and fills a new `BinaryVector`. For dimension `i`: bit is `1` if `vec[i] > thresholds[i]`, else `0`. Returns an error if `len(vec) != len(thresholds)` — a sign the embedder dimension changed and the binary index is stale.

### `QuantizeInto(dst BinaryVector, vec, thresholds []float32) error`

Encodes into an existing buffer, clearing bits before writing. Returns an error if `len(vec) != len(thresholds)` or `len(dst) != BinaryWords(len(vec))`. Use when amortizing allocations across many calls.

### `HammingDistance(a, b BinaryVector) int`

Number of differing bits between two binary vectors. Returns `0` if `len(a) != len(b)`.

### `ComputeMedians(vecs [][]float32) []float32`

Per-dimension median across a set of vectors. Suitable as the `thresholds` argument to `Quantize`. Returns `nil` if `vecs` is empty; returns a copy for a single-vector input. Silently returns `nil` on dimension mismatch — use `ComputeMediansChecked` when callers must distinguish an error from an empty input.

### `ComputeMediansChecked(vecs [][]float32) ([]float32, error)`

Validated form of `ComputeMedians`. Returns an error on dimension mismatch between vectors.

---

## near-duplicate detection

Group or pair embeddings that are close in cosine space. This is the **semantic** complement to the lexical SimHash near-dup primitive in the `dedup` module: SimHash catches near-identical surface text; these functions catch vectors close in embedding space regardless of wording.

```go
groups := vectors.NearDuplicateGroups(corpus, 0.92) // connected components, size >= 2
pairs  := vectors.NearDuplicatePairs(corpus, 0.92)  // raw linked (i<j) index pairs
```

v1 verifies every pair with exact `Cosine32`, so the result is identical to a brute-force `Cosine32` scan — no true positive is dropped. Binary quantization (`ComputeMedians` + `Quantize`) is wired in as the O(n) screen a future radius-bounded prefilter would consume; median-threshold quantization admits no safe tight Hamming↔cosine bound, so v1 does not prune by Hamming radius.

### `NearDuplicateGroups(vecs [][]float32, cosineThreshold float32) [][]int`

Groups vectors into clusters of mutual near-duplicates. Two vectors are linked when `Cosine32 >= cosineThreshold`; groups are the connected components of that graph with singletons omitted. Returns indices into `vecs`, each group sorted ascending, groups ordered by smallest member. `nil`/empty input or fewer than two usable vectors returns an empty result; `nil`/zero-length member vectors are skipped and never linked.

### `NearDuplicatePairs(vecs [][]float32, cosineThreshold float32) [][2]int`

Returns the linked index pairs `(i<j, Cosine32>=cosineThreshold)` in ascending `(i, j)` order rather than connected-component groups. Same guards and member-skipping rules as `NearDuplicateGroups`.

---

## encoding (BLOB)

These functions convert between `[]float32`/`[]float64` slices and raw little-endian byte blobs for storage (e.g. SQLite `BLOB` columns).

```go
blob := vectors.EncodeFloat32Vec(embedding)        // store to DB
vec  := vectors.DecodeFloat32Vec(blob)             // load from DB; nil on bad length

// compute cosine without a full decode:
score := vectors.DotFromBlob(queryVec, blob)        // both must be L2-normalized
```

### `EncodeFloat32Vec(v []float32) []byte`

Encodes `v` as a raw little-endian byte slice. Each float32 occupies 4 bytes (512 dims → 2048 bytes).

### `EncodeFloat64Vec(v []float64) []byte`

Encodes `v` as a raw little-endian byte slice. Each float64 occupies 8 bytes. Returns `nil` for a `nil` input.

### `DecodeFloat32Vec(blob []byte) []float32`

Decodes a little-endian float32 blob. Returns `nil` if `len(blob)` is not a multiple of 4.

### `DecodeFloat64Vec(data []byte) []float64`

Decodes a little-endian float64 blob. Returns `nil` if `len(data)` is not a multiple of 8. Returns an empty (non-nil) slice for empty input.

### `DecodeFloat32Batch(blobs [][]byte) ([][]float32, error)`

Decodes a slice of float32 blobs preserving input order (`out[i]` corresponds to `blobs[i]`). Fails fast with an error naming the first blob whose length is not a multiple of 4.

### `DotFromBlob(query []float32, blob []byte) float32`

Computes the dot product between a pre-normalized query vector and a raw little-endian float32 BLOB without allocating an intermediate slice. **Both operands must be L2-normalized and contain only finite values**: under that invariant, dot product equals cosine similarity. This is a low-level precondition-based primitive; non-finite inputs may produce non-finite scores. Returns `0` if dimensions do not match.

### `type ScoredIndex struct`

```go
type ScoredIndex struct {
    Index int
    Score float32
}
```

### `TopKFromBlob(query []float32, blobs [][]byte, limit int, minScore float32) []ScoredIndex`

Scores raw little-endian float32 blobs against `query` and returns the top input indexes. Invalid or dimension-mismatched blobs and blobs producing non-finite scores are skipped. Equal scores are ordered by ascending input index. Returns an empty slice for empty or non-finite queries, empty blob lists, or `limit <= 0`.

---

## scoring & fusion

### `Clamp01(x float64) float64`

Clamps `x` to `[0, 1]`. NaN and negative values collapse to `0`; values above `1` collapse to `1`. NaN check runs before range comparison so NaN cannot propagate through a downstream weighted sum.

### `CosineToUnit(score float64) float64`

Remaps a cosine similarity in `[-1, 1]` to `[0, 1]` via `(score+1)/2`. NaN input maps to `0` before the arithmetic. Result is `Clamp01`-guarded against out-of-range cosine inputs.

### `type Scored struct { Index int; Score float64 }`

Pairs an index with its fused score. Sorted descending by score, then ascending by index for deterministic tie-breaks.

### `RRF(ranked [][]int, k float64) ([]Scored, float64)`

Reciprocal Rank Fusion over multiple ranked index lists. Each list is a ranking where position `i` has rank `i+1`; each contribution is `1/(k+rank)`. Raw metric scores are not used — only rank positions. Returns the fused ranking (descending score, ascending index on ties) and the maximum score. `k ≤ 0` uses the Cormack et al. (2009) default of 60. Returns `(nil, 0)` for empty input.

### `MeanStddev(vals []float64) (mean, stddev float64)`

Mean and population standard deviation (divisor N, not N-1) of `vals`. Empty input returns `(0, 0)`.

### `MinMaxNormalize(vals []float64) []float64`

Rescales `vals` to `[0, 1]` via `(v−min)/(max−min)`. Returns all `0.5` when fewer than 2 elements exist or when `max−min < 1e-10` (degenerate spread avoids NaN propagation). Empty input returns an empty non-nil slice.

---

## pooling

### `MeanPool32(vecs [][]float32) []float32`

Averages a set of float32 vectors componentwise and L2-normalizes the result. Members whose length differs from the first vector's length are skipped. Edge cases:
- empty input → `nil`
- single-vector input → the input slice, **not** normalized (caller must normalize if needed)

### `WeightedMeanPool32(vecs [][]float32, weights []float64) ([]float32, error)`

Weighted mean of vectors, L2-normalized. Returns an error on weight count mismatch, ragged vectors, or invalid weights (negative, NaN, Inf). Zero-weight vectors are skipped. Returns a zero vector (not `nil`) when all weights sum to zero.

---

## selection

### `TopKMaxIndices(scores []float32, k int) []int`

Indices of the `k` largest values in `scores`, ordered largest-first. `k` is clamped to `len(scores)`. `k ≤ 0` returns an empty slice. Equal scores are ordered by ascending index (stable). Runs in `O(n log k)` time, `O(k)` space.

### `TopKMinIndices(scores []float32, k int) []int`

Indices of the `k` smallest values in `scores`, ordered smallest-first. Same clamping and stability rules as `TopKMaxIndices`.

---

## indexes

Two in-memory indexes cover different points in the latency/accuracy trade-off:

- **`ExactIndex`** — full brute-force cosine search over L2-normalized float32 vectors packed in a single byte arena.
- **`BinaryIndex`** — Hamming-distance pre-filter for candidate generation. Pair with `ExactIndex.SearchKeys` for chunk-level exact reranking or `ExactIndex.SearchFiltered` for entry-level filtering.

### ExactIndex

```go
// Build once; search many times.
idx := vectors.NewExactIndex(dims, chunks, arena)

results, ok := idx.Search(queryVec, 10, 0.7)
for _, r := range results {
    fmt.Println(r.Group, r.ChunkIndex, r.Score)
}
```

#### `type IndexChunk struct`

```go
type IndexChunk struct {
    Group      string // e.g. entry ID
    ChunkIndex int    // 0-based chunk offset within the group
    Offset     int    // byte offset into the arena slice
    Length     int    // byte length of this vector in the arena
}
```

#### `type SearchResult struct`

```go
type SearchResult struct {
    Group      string
    ChunkIndex int
    Score      float64
}
```

#### `type IndexKey struct`

```go
type IndexKey struct {
    Group      string
    ChunkIndex int
}
```

Identifies one indexed chunk without tying `vectors` to caller storage. `BinaryIndex` candidates can be converted directly into `IndexKey` values for exact reranking.

#### `type IndexBuildReport struct`

```go
type IndexBuildReport struct {
    InputRows              int
    IndexedRows            int
    SkippedBadSpan         int
    SkippedBadBlob         int
    SkippedMissingMetadata int
    SkippedDuplicateKey    int
    DimensionMismatch      int
    MedianError            string
    QuantizeError          string
}
```

Reports recoverable index-build problems. `SkippedBadBlob` includes encoded vectors containing NaN or infinity. Fields that do not apply to an index type remain zero.

#### `NewExactIndex(dims int, chunks []IndexChunk, arena []byte) *ExactIndex`

Constructs an `ExactIndex`. `arena` is a single contiguous byte slice containing all float32 vectors encoded with `EncodeFloat32Vec`; `chunks` describes each vector's position within it. The constructor snapshots the complete arena, so the caller may mutate or release its slice after return. Builds an internal group→range index for filtered search. A non-positive `dims` value produces a non-nil empty index.

#### `NewExactIndexChecked(dims int, chunks []IndexChunk, arena []byte) (*ExactIndex, IndexBuildReport)`

Constructs an `ExactIndex` and returns a build report. Out-of-bounds arena spans are skipped. In-bounds rows with lengths that do not match `dims*4` are skipped and reported via `DimensionMismatch`. When `dims` is non-positive, every otherwise valid row is reported via `DimensionMismatch` and the returned index is non-nil and empty. Rows containing NaN or infinity are skipped and reported via `SkippedBadBlob` before duplicate keys are reserved. For duplicate `(Group, ChunkIndex)` keys, the first valid row is retained and later valid rows are skipped and reported via `SkippedDuplicateKey`. The constructor snapshots the complete arena, so the caller may mutate or release its slice after return.

#### `(*ExactIndex).Len() int`

Number of chunks stored. Thread-safe.

#### `(*ExactIndex).Clear()`

Clears all internal slices and maps, releasing memory. Thread-safe.

#### `(*ExactIndex).Search(queryVec []float32, limit int, minSimilarity float64) ([]SearchResult, bool)`

Scores all vectors against `queryVec` and returns the top-`limit` matches. `limit ≤ 0` defaults to 10. Equal scores are ordered by `Group`, then `ChunkIndex`. Returns `(nil, false)` when the index is empty or the query has the wrong dimension or contains NaN or infinity.

#### `(*ExactIndex).SearchFiltered(queryVec []float32, limit int, minSimilarity float64, groups []string) ([]SearchResult, bool)`

Scores only vectors belonging to the specified groups. Equal scores are ordered by `Group`, then `ChunkIndex`. Returns `(nil, false)` when the index is empty or the query has the wrong dimension or contains NaN or infinity; returns `(nil, true)` when none of the groups exist.

#### `(*ExactIndex).SearchKeys(queryVec []float32, limit int, minSimilarity float64, keys []IndexKey) ([]SearchResult, bool)`

Scores only chunks matching the supplied keys. Duplicate and missing keys are ignored. Equal scores are ordered by `Group`, then `ChunkIndex`. Returns `(nil, false)` when the index is empty or the query has the wrong dimension or contains NaN or infinity. This is the preferred exact rerank primitive for two-stage pipelines where `BinaryIndex.SearchCandidates` has already produced candidate `(Group, ChunkIndex)` pairs.

#### `(*ExactIndex).SearchGroupsByMaxPool(queryVec []float32, limit int) ([]SearchResult, bool)`

Pools the best similarity score per group across all chunks and returns the top-`limit` best matching groups. `ChunkIndex` is `-1` in results (group-level, not chunk-level), and equal scores are ordered by `Group`. Returns `(nil, false)` when the index is empty or the query has the wrong dimension or contains NaN or infinity.

---

### BinaryIndex

```go
// Two-stage ANN: fast Hamming pre-filter → exact re-rank.
binIdx := vectors.NewBinaryIndex(blobs, groups, chunkIndices, dims)

candidates, ok := binIdx.SearchCandidates(queryVec)
// candidates are sorted by ascending Hamming distance (most similar first).
keys := make([]vectors.IndexKey, len(candidates))
for i, c := range candidates {
    keys[i] = vectors.IndexKey{Group: c.Group, ChunkIndex: c.ChunkIndex}
}
results, ok := exactIdx.SearchKeys(queryVec, 10, 0.7, keys)
```

#### `type BinaryIndexEntry struct`

```go
type BinaryIndexEntry struct {
    Group      string
    ChunkIndex int
}
```

#### `type HammingCandidate struct`

```go
type HammingCandidate struct {
    Group      string
    ChunkIndex int
    Hamming    int // lower = more similar
}
```

#### `NewBinaryIndex(blobs [][]byte, groups []string, chunkIndices []int, dims int) *BinaryIndex`

Builds a `BinaryIndex` from raw little-endian float32 blobs. Decodes each blob, computes per-dimension medians across all vectors, then binary-quantizes every vector. Blobs with invalid lengths, NaN or infinity, dimension mismatches, and rows missing group/chunk metadata are skipped. Duplicate `(Group, ChunkIndex)` keys retain the first valid row. Returns an empty index if no valid rows are provided.

#### `NewBinaryIndexChecked(blobs [][]byte, groups []string, chunkIndices []int, dims int) (*BinaryIndex, IndexBuildReport)`

Builds a `BinaryIndex` and returns a report with skipped blob, missing metadata, duplicate key, dimension mismatch, median, and quantization counters/errors. Rows are validated before reserving their key, so an invalid row does not suppress a later valid row with the same key.

#### `(*BinaryIndex).SearchCandidates(queryVec []float32) ([]HammingCandidate, bool)`

Quantizes `queryVec` using the stored medians and returns the top-100 candidates by ascending Hamming distance. This is a compatibility wrapper around `SearchCandidatesLimit(queryVec, 100)`. Returns `(nil, false)` when the index is empty, the query contains NaN or infinity, or quantization fails.

#### `(*BinaryIndex).SearchCandidatesLimit(queryVec []float32, limit int) ([]HammingCandidate, bool)`

Quantizes `queryVec` using the stored medians and returns up to `limit` candidates by ascending Hamming distance. `limit` is clamped to the index size; `limit ≤ 0` returns an empty candidate slice for a valid query. Returns `(nil, false)` when the index is empty, the query contains NaN or infinity, or quantization fails. Equal Hamming distances are ordered by group, then chunk index, so cutoff ties are deterministic.

#### `(*BinaryIndex).Len() int`

Number of entries. Thread-safe.

#### `(*BinaryIndex).Clear()`

Clears all internal state. Thread-safe.

---

## semantic windowing & curvature

These functions detect topic boundaries and score curves in sequences of sentence embeddings, used by the `chunking` package.

```go
sims := vectors.SlidingWindowSimilarity(embeddings, 3)
sims  = vectors.SmoothSimilarities(sims, 3)
threshold := vectors.AdaptiveThreshold(sims)
boundaries := vectors.FindSemanticBoundaries(sims, threshold)
```

### `AverageSimilarity(embeddings [][]float32) float32`

Average pairwise cosine similarity across all pairs. Returns `1` when fewer than two embeddings are provided.

### `SlidingWindowSimilarity(embeddings [][]float32, windowSize int) []float32`

Similarity scores between consecutive sliding windows over `embeddings`. Returns `len(embeddings)-1` scores. Returns an empty slice when `len(embeddings) < 2` or `windowSize < 1`.

### `FindSemanticBoundaries(similarities []float32, threshold float32) []int`

Returns 1-based positions where similarity drops below `threshold`. A boundary at position `i+1` means the gap between window `i` and `i+1` is a topic change.

### `AdaptiveThreshold(similarities []float32) float32`

Computes `mean − stddev` of `similarities`, clamped to `[0.3, 0.9]`. Returns `0.7` for empty input.

### `SmoothSimilarities(similarities []float32, windowSize int) []float32`

Box-car smoothing with a window of `windowSize`. Returns the input unchanged when `len(similarities) == 0` or `windowSize < 1`.

### `JaccardWords(text1, text2 string) float64`

Jaccard similarity between the meaningful words in two texts. Strips code fences, URLs, and common stop words before comparison. Returns `0` for empty inputs.

### `GaussianSmooth(scores []float32, sigma float64) []float32`

1D Gaussian kernel convolution over `scores`. `sigma` controls smoothing width; a typical value is `1.0`. Edges are clamped (not zero-padded). Returns a copy unchanged for inputs shorter than 2 or invalid sigma values (`<= 0`, `NaN`, or infinity).

### `Gradient(scores []float32) []float32`

Discrete gradient using central differences, forward difference at index 0, backward at index `n−1`. Returns a zero-length slice for empty input.

### `FindElbowCurvature(scores []float32, minKeep int) int`

Returns the index at which the smoothed curvature (`|d²/di²|` of the Gaussian-smoothed `scores`) peaks. The search starts at `minKeep` so no cut point before that index is returned. Returns `len(scores)-1` when no meaningful peak is found (flat or trivially short input).

---

## in-package clustering (float32)

These functions cluster float32 embeddings directly. For new float64 workflows,
use the `clustering` sub-package; the root package also retains the compatibility
API documented below.

**Points must be non-empty, non-zero-dimensional, uniform-dimensional, finite, and L2-normalized** before passing to `KMeans` or `SilhouetteScore`.

```go
rng := rand.New(rand.NewSource(42))
result := vectors.KMeans(embeddings, 5, rng)

score := vectors.SilhouetteScore(embeddings, result.Assignments, result.K)

bestK, bestScore := vectors.FindOptimalK(embeddings, 2, 10, rng)
```

### `type KMeansResult struct`

```go
type KMeansResult struct {
    K           int
    Assignments []int       // Assignments[i] = cluster ID for points[i]
    Centroids   [][]float32 // Centroids[j] = centroid of cluster j
    Iterations  int
}
```

### `KMeans(points [][]float32, k int, rng *rand.Rand) *KMeansResult`

K-means clustering with K-means++ initialization using cosine distance (spherical K-means). `k` is clamped to `len(points)`. A nil RNG uses a deterministic fallback. Ragged points, zero-dimensional points, and points containing NaN or infinity return an empty result. Uses a fixed convergence tolerance of `1e-6` and a maximum of 100 iterations. Empty clusters are recovered by reinitializing from the point furthest from its assigned centroid.

### `SilhouetteScore(points [][]float32, assignments []int, k int) float64`

Average silhouette coefficient for the given clustering. For datasets larger than 1000 points, uses a deterministic subsample (seeded by `n*k`) to keep computation tractable. Returns `0` when `n < 2` or `k < 2`.

### `FindOptimalK(points [][]float32, minK, maxK int, rng *rand.Rand) (int, float64)`

Runs `KMeans` for each `k` in `[minK, maxK]` and returns the `k` with the highest silhouette score. Returns `(minK, score)` when all silhouette scores are equal.

### `KMeans64(points [][]float64, cfg KMeans64Config) *KMeans64Result`

Float64 K-means with K-means++ initialization and cosine distance. Non-positive `K`, `MaxIterations`, and `Tolerance` use defaults of 2, 100, and `1e-4`; `K` is capped at the point count. `Seed == 0` selects a deterministic seed derived from the input size and `K`. Empty, ragged, zero-dimensional, or non-finite points return an empty result with `K == 0`.

```go
type KMeans64Config struct {
    K             int
    MaxIterations int
    Tolerance     float64
    Seed          int64
}

type KMeans64Result struct {
    Assignments []int
    Centroids   [][]float64
    K           int
    Iterations  int
    Converged   bool
}
```

### `FindOptimalK64(points [][]float64, minK, maxK int) (bestK int, bestScore float64, assignments []int, centroids [][]float64, scores []float64, kValues []int)`

Evaluates each feasible `k` with float64 K-means and silhouette scoring. Malformed points or a candidate range with no feasible `k` return `(0, 0, nil, nil, nil, nil)`. With a non-inverted candidate range, valid one- or two-point inputs preserve the legacy result: `bestK == 1`, `bestScore == 0`, all assignments are zero, `centroids == nil`, `scores == []float64{0}`, and `kValues == []int{1}`.

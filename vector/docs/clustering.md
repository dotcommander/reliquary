# api reference — `github.com/dotcommander/reliquary/vector/clustering`

```go
import "github.com/dotcommander/reliquary/vector/clustering"

svc := clustering.NewClusterService("kmeans")
opts := clustering.DefaultClusterOptions()
opts.K = 5

result, err := svc.Cluster(embeddings, opts)
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.K, result.Silhouette)
```

The `clustering` sub-package operates on **float64 embeddings** and provides three algorithms — greedy similarity grouping, K-means, and hierarchical agglomerative clustering (HAC) — behind a single `ClusterService` interface. It depends on the root `vectors` package for distance primitives.

---

## ClusterService interface

### `NewClusterService(algorithm string) ClusterService`

Creates a `ClusterService` backed by the named algorithm. Supported values: `"greedy"` (default), `"kmeans"`, `"hac"`. An empty string defaults to `"greedy"`.

### `type ClusterService interface`

```go
type ClusterService interface {
    Cluster(embeddings [][]float64, opts ClusterOptions) (*ClusterResult, error)
}
```

### `type ClusterOptions struct`

```go
type ClusterOptions struct {
    Algorithm string  // "greedy", "kmeans", "hac"
    K         int     // 0 = auto-select via silhouette
    MaxK      int     // max k for auto-selection (default: 20)
    Linkage   string  // for HAC: "single", "complete", "average"

    // greedy-specific
    TargetMax int     // target items per cluster (default: 8)
    HardMax   int     // hard cap on cluster size (default: 12)
    MaxSizeKB int     // max total size in KB per cluster (default: 80)
    FileSizes []int64 // per-item sizes in bytes (for greedy size constraints)
    Threshold float64 // minimum similarity threshold for greedy (0 = disabled)
}
```

### `DefaultClusterOptions() ClusterOptions`

```go
ClusterOptions{
    Algorithm: "greedy",
    K:         0,
    MaxK:      20,
    Linkage:   "average",
    TargetMax: 8,
    HardMax:   12,
    MaxSizeKB: 80,
    Threshold: 0.0,
}
```

### `type ClusterResult struct`

```go
type ClusterResult struct {
    Assignments []int       // cluster ID for each embedding
    K           int         // number of clusters
    Centroids   [][]float64 // cluster centroids (for kmeans and HAC)
    Silhouette  float64     // average silhouette score
}
```

`Cluster` returns an empty `ClusterResult` (no error) for empty input.

---

## algorithms

### greedy

The greedy algorithm groups embeddings in similarity order. It picks the first unassigned embedding as a seed, scores all remaining unassigned embeddings by cosine similarity to the seed, then greedily adds the most similar candidates while respecting `TargetMax`, `HardMax`, and `MaxSizeKB` limits. It repeats until all embeddings are assigned.

Use greedy when you need bounded cluster sizes (e.g. batching files into LLM context windows) rather than globally optimal clusters.

**Relevant fields**: `TargetMax`, `HardMax`, `MaxSizeKB`, `FileSizes`, `Threshold`.

---

### K-means

```go
svc := clustering.NewClusterService("kmeans")
opts := clustering.DefaultClusterOptions()
opts.Algorithm = "kmeans"
opts.K = 0          // 0 = auto-select via silhouette from 2..MaxK
opts.MaxK = 15

result, _ := svc.Cluster(embeddings, opts)
```

When `K == 0`, the service calls `FindOptimalK` internally to sweep `k` from 2 to `MaxK` and picks the best silhouette score. When `K > 0`, a single `KMeans` run is performed.

#### `type KMeansConfig struct`

```go
type KMeansConfig struct {
    K             int     // number of clusters (0 = K-means++ selects 2)
    MaxIterations int     // default: 100
    Tolerance     float64 // convergence tolerance (default: 1e-4)
    Seed          int64   // 0 = random seed
}
```

#### `DefaultKMeansConfig() KMeansConfig`

```go
KMeansConfig{
    K:             0,
    MaxIterations: 100,
    Tolerance:     1e-4,
    Seed:          0,
}
```

#### `KMeans(embeddings [][]float64, cfg KMeansConfig) *KMeansResult`

K-means clustering with K-means++ initialization. Uses cosine distance. `K ≤ 0` defaults to 2. Returns an empty result for empty input.

#### `type KMeansResult struct`

```go
type KMeansResult struct {
    Assignments []int       // cluster ID for each point
    Centroids   [][]float64 // cluster centroids
    K           int         // number of clusters
    Iterations  int         // iterations until convergence
    Converged   bool        // whether algorithm converged within MaxIterations
}
```

---

### HAC (hierarchical agglomerative clustering)

```go
svc := clustering.NewClusterService("hac")
opts := clustering.DefaultClusterOptions()
opts.Algorithm = "hac"
opts.K = 4
opts.Linkage = "average"

result, _ := svc.Cluster(embeddings, opts)
```

HAC merges the closest pair of clusters at each step until `K` clusters remain, recording the full merge history as a dendrogram.

#### `type Linkage string`

```go
const (
    LinkageSingle   Linkage = "single"   // minimum distance between clusters
    LinkageComplete Linkage = "complete" // maximum distance between clusters
    LinkageAverage  Linkage = "average"  // average distance between clusters
)
```

#### `type HACConfig struct`

```go
type HACConfig struct {
    K       int     // target number of clusters (0 defaults to 2)
    Linkage Linkage // default: average
}
```

#### `DefaultHACConfig() HACConfig`

```go
HACConfig{
    K:       0,
    Linkage: LinkageAverage,
}
```

#### `HAC(embeddings [][]float64, cfg HACConfig) *HACResult`

Hierarchical agglomerative clustering using cosine distance. Precomputes the full distance matrix (O(n²) memory). `K ≤ 0` defaults to 2. Returns an empty result for empty input.

#### `type HACResult struct`

```go
type HACResult struct {
    Assignments []int       // cluster ID for each point
    Centroids   [][]float64 // cluster centroids
    K           int         // number of clusters
    Dendrogram  []MergeStep // merge history
}
```

#### `type MergeStep struct`

```go
type MergeStep struct {
    ClusterA int     // first cluster merged
    ClusterB int     // second cluster merged
    Distance float64 // distance at merge
    NewSize  int     // size of merged cluster
}
```

#### `CutDendrogram(dendrogram []MergeStep, n int, distanceThreshold float64) []int`

Cuts the dendrogram at a distance threshold: merges below the threshold are applied, producing cluster assignments. Returns contiguously renumbered assignments starting from 0. `n` must equal the number of original points.

---

## silhouette analysis & auto-k selection

The silhouette subsystem drives auto-k selection when `K == 0` in `ClusterOptions`. You can also call it directly to evaluate an existing clustering or to select k independent of the service.

```go
// Evaluate an existing clustering.
score := clustering.AverageSilhouetteScore(embeddings, assignments)

// Per-cluster scores — identify weak clusters.
perCluster := clustering.ClusterSilhouetteScores(embeddings, assignments)

// Select k automatically.
cfg := clustering.DefaultSilhouetteConfig()
cfg.MaxK = 12
cfg.Algorithm = "kmeans"
result := clustering.FindOptimalK(embeddings, cfg)
fmt.Println(result.BestK, result.BestScore)
```

### `type SilhouetteConfig struct`

```go
type SilhouetteConfig struct {
    MinK      int    // default: 2
    MaxK      int    // default: 20
    Algorithm string // "kmeans" or "hac" (default: "kmeans")
}
```

### `DefaultSilhouetteConfig() SilhouetteConfig`

```go
SilhouetteConfig{
    MinK:      2,
    MaxK:      20,
    Algorithm: "kmeans",
}
```

### `type SilhouetteResult struct`

```go
type SilhouetteResult struct {
    BestK       int         // best k (tie-break: smaller k)
    BestScore   float64     // silhouette score at best k
    Scores      []float64   // silhouette scores for each k tried
    KValues     []int       // k values tried
    Assignments []int       // cluster assignments at best k
    Centroids   [][]float64
}
```

### `SilhouetteCoefficient(pointIdx int, embeddings [][]float64, assignments []int) float64`

Silhouette coefficient for a single point. Returns a value in `[-1, 1]`:
- `a` = average cosine distance to other points in the same cluster
- `b` = average cosine distance to points in the nearest other cluster
- coefficient = `(b - a) / max(a, b)`

Returns `0` for single-point clusters or when only one cluster exists. Returns `0` when `len(embeddings) ≤ 1`.

### `AverageSilhouetteScore(embeddings [][]float64, assignments []int) float64`

Mean silhouette coefficient across all points. Returns `0` for empty input.

### `FindOptimalK(embeddings [][]float64, cfg SilhouetteConfig) *SilhouetteResult`

Sweeps `k` from `cfg.MinK` to `min(cfg.MaxK, n-1)` and returns the best k by silhouette score. Tie-break: smaller k wins (simpler model). Returns `BestK=1` with a zero score when `n ≤ 2`.

### `ClusterSilhouetteScores(embeddings [][]float64, assignments []int) map[int]float64`

Per-cluster average silhouette score. Useful for identifying poorly separated clusters. Returns an empty map for empty input.

---

## compat helpers

These functions delegate to the root `vectors` package and are exposed so that HAC internals and user code can work with `float64` slices without importing both packages.

### `CosineDistance(a, b []float64) float64`

Delegates to `vectors.CosineDistance64`. Returns `1 - cosine(a, b)`.

### `EuclideanDistance(a, b []float64) float64`

Delegates to `vectors.Euclidean64`. L2 distance.

### `ComputeCentroid(points [][]float64) []float64`

Delegates to `vectors.ComputeCentroid64`. Mean vector of `points`.

### `type DistanceFunc func(a, b []float64) float64`

Function type for a distance metric. Passed to `DistanceMatrix` and available for custom HAC usage.

### `DistanceMatrix(points [][]float64, metric DistanceFunc) [][]float64`

Symmetric pairwise distance matrix using `metric`. Returns `nil` for empty input. Computes each pair once: `O(n²/2)` metric calls, `O(n²)` memory.

### `NormalizeVector(vec []float64) []float64`

Normalizes `vec` in place via `vectors.Normalize64` and returns it. Returns the input unchanged for empty input.

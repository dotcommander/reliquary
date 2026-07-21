# vectors/clustering

```go
service := clustering.NewClusterService("kmeans")
result, err := service.Cluster(vectors, clustering.ClusterOptions{K: 2})
```

`vectors/clustering` groups caller-supplied vectors with greedy, k-means, and
hierarchical agglomerative clustering helpers.

`Cluster` accepts empty input as a successful empty result. Non-empty input
must have a positive, consistent dimension and contain only finite values;
malformed input returns an error before any algorithm runs.

It never embeds text or chooses model identity. Callers own vector-space
consistency, source documents, labels, and downstream policy.

# vectors/clustering

```go
service := clustering.NewClusterService("kmeans")
result, err := service.Cluster(vectors, clustering.ClusterOptions{K: 2})
```

`vectors/clustering` groups caller-supplied vectors with greedy, k-means, and
hierarchical agglomerative clustering helpers.

It never embeds text or chooses model identity. Callers own vector-space
consistency, source documents, labels, and downstream policy.

// Package cache decorates an embedding.Embedder with an explicit, caller-owned
// Store. It caches individual input vectors while preserving embedding batch
// order and resolved model identity.
package cache

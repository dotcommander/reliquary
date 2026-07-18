// Package embeddings defines provider-neutral embedding contracts.
package embeddings

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

// Vector is a dense embedding vector.
type Vector []float32

// ModelRef identifies an embedding model and vector space.
type ModelRef struct {
	Provider string
	Name     string
	Version  string
	Revision string
	Dim      int
}

// Identity returns the stable model identity used for cache keys.
func (m ModelRef) Identity() string {
	parts := []string{m.Provider, m.Name, m.Version, m.Revision, fmt.Sprint(m.Dim)}
	return strings.Join(parts, ":")
}

// Request is a batch embedding request.
type Request struct {
	Model  ModelRef
	Inputs []string
}

// Result is a batch embedding result.
type Result struct {
	Model   ModelRef
	Vectors []Vector
}

// Embedder embeds text into vectors.
type Embedder interface {
	Embed(ctx context.Context, request Request) (Result, error)
}

// ValidateDimensions checks that every vector has dims values.
func ValidateDimensions(vectors []Vector, dims int) error {
	if dims <= 0 {
		return fmt.Errorf("embeddings: invalid dims %d", dims)
	}
	for i, vector := range vectors {
		if len(vector) != dims {
			return fmt.Errorf("embeddings: vector %d has dims %d, expected %d", i, len(vector), dims)
		}
	}
	return nil
}

// CacheKey returns a stable cache identity for one model/input pair.
func CacheKey(model ModelRef, input string) string {
	sum := sha256.Sum256([]byte(model.Identity() + "\x00" + input))
	return hex.EncodeToString(sum[:])
}

// Package embed provides a deterministic, dependency-free embedding.Embedder
// for demos, tests, and reliquary.Quickstart. It maps text to vectors with the
// signed feature-hashing trick, so callers obtain meaningful (non-trivial
// cosine) vectors without an ONNX runtime or API key.
//
// It is a stand-in for a real embedding model, not a replacement: quality is
// suitable for examples and tests only, never production retrieval.
package embed

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"unicode"

	"github.com/dotcommander/reliquary/embedding"
)

// DefaultHashingDim is used when callers pass a non-positive dimension. It is
// intentionally small and deterministic for demos and tests, not production
// retrieval quality.
const DefaultHashingDim = 256

// Hashing is a deterministic hashing-trick embedder implementing
// embedding.Embedder.
type Hashing struct {
	Model embedding.ModelRef
}

// NewHashing returns a Hashing embedder producing L2-normalized vectors of the
// given width. Non-positive dimensions use DefaultHashingDim.
func NewHashing(dim int) *Hashing {
	if dim <= 0 {
		dim = DefaultHashingDim
	}
	return &Hashing{Model: embedding.ModelRef{Provider: "demo", Name: "hashing", Version: "1", Dim: dim}}
}

// Embed satisfies embedding.Embedder.
func (h *Hashing) Embed(ctx context.Context, req embedding.Request) (embedding.Result, error) {
	if err := ctx.Err(); err != nil {
		return embedding.Result{}, err
	}
	vectors := make([]embedding.Vector, len(req.Inputs))
	for i, in := range req.Inputs {
		vectors[i] = HashVector(in, h.Model.Dim)
	}
	return embedding.Result{Model: h.Model, Vectors: vectors}, nil
}

// HashVector maps text to an L2-normalized []float32 via signed feature hashing.
// Non-positive dimensions use DefaultHashingDim.
func HashVector(text string, dim int) embedding.Vector {
	if dim <= 0 {
		dim = DefaultHashingDim
	}
	v := make(embedding.Vector, dim)
	for _, tok := range tokenize(text) {
		hsh := fnv1a64(tok)
		idx := int(hsh % uint64(dim))
		sign := float32(1)
		if (hsh>>63)&1 == 1 {
			sign = -1
		}
		v[idx] += sign
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum > 0 {
		norm := math.Sqrt(sum)
		for i := range v {
			v[i] = float32(float64(v[i]) / norm)
		}
	}
	return v
}

func fnv1a64(s string) uint64 {
	f := fnv.New64a()
	_, _ = f.Write([]byte(s))
	return f.Sum64()
}

func tokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(unicode.IsLetter(r) || unicode.IsDigit(r))
	})
}

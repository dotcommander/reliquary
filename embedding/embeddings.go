// Package embeddings defines provider-neutral embedding contracts.
package embeddings

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ErrInvalidResult indicates that an embedder returned a malformed result.
var ErrInvalidResult = errors.New("embeddings: invalid result")

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

// Identity returns the versioned, byte-length-framed model identity used for
// cache keys. Format changes intentionally invalidate prior cache entries.
func (m ModelRef) Identity() string {
	fields := [...]string{m.Provider, m.Name, m.Version, m.Revision, strconv.Itoa(m.Dim)}
	var identity strings.Builder
	identity.WriteString("modelref:v1:")
	for _, field := range fields {
		identity.WriteString(strconv.Itoa(len(field)))
		identity.WriteByte(':')
		identity.WriteString(field)
	}
	return identity.String()
}

// Request is a batch embedding request.
type Request struct {
	Model  ModelRef
	Inputs []string
}

// Result is a batch embedding result. A successful Embed call returns exactly
// one vector per input, in the same order as the request inputs.
type Result struct {
	Model   ModelRef
	Vectors []Vector
}

// Embedder embeds text into vectors. Successful results contain exactly one
// vector per input, in the same order as the request inputs.
type Embedder interface {
	Embed(ctx context.Context, request Request) (Result, error)
}

// ValidateResult verifies the shape and finite values required of a successful
// embedding result. A zero ModelRef.Dim means unspecified; when present,
// request and result dimensions must agree with each other and every vector.
func ValidateResult(request Request, result Result) error {
	if request.Model.Dim < 0 {
		return fmt.Errorf("%w: request model has invalid dims %d", ErrInvalidResult, request.Model.Dim)
	}
	if result.Model.Dim < 0 {
		return fmt.Errorf("%w: result model has invalid dims %d", ErrInvalidResult, result.Model.Dim)
	}
	if len(result.Vectors) != len(request.Inputs) {
		return fmt.Errorf("%w: got %d vectors for %d inputs", ErrInvalidResult, len(result.Vectors), len(request.Inputs))
	}
	if request.Model.Dim > 0 && result.Model.Dim > 0 && request.Model.Dim != result.Model.Dim {
		return fmt.Errorf("%w: request dims %d do not match result dims %d", ErrInvalidResult, request.Model.Dim, result.Model.Dim)
	}
	if len(result.Vectors) == 0 {
		return nil
	}

	dims := len(result.Vectors[0])
	if dims == 0 {
		return fmt.Errorf("%w: vector 0 has invalid dims 0", ErrInvalidResult)
	}
	if request.Model.Dim > 0 && request.Model.Dim != dims {
		return fmt.Errorf("%w: vector dims %d do not match request dims %d", ErrInvalidResult, dims, request.Model.Dim)
	}
	if result.Model.Dim > 0 && result.Model.Dim != dims {
		return fmt.Errorf("%w: vector dims %d do not match result dims %d", ErrInvalidResult, dims, result.Model.Dim)
	}
	for i, vector := range result.Vectors {
		if len(vector) != dims {
			return fmt.Errorf("%w: vector %d has dims %d, expected %d", ErrInvalidResult, i, len(vector), dims)
		}
		for j, value := range vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("%w: vector %d value %d is not finite", ErrInvalidResult, i, j)
			}
		}
	}
	return nil
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

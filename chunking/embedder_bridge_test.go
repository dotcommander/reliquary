package chunking

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/dotcommander/reliquary/embedding"
)

// recordingEmbedder is a deterministic embedding.Embedder fake: it returns one
// vector per input, tagged by input index, and records every request.
type recordingEmbedder struct {
	mu        sync.Mutex
	requests  []embedding.Request
	err       error
	shortByID bool
}

func (r *recordingEmbedder) Embed(_ context.Context, req embedding.Request) (embedding.Result, error) {
	if r.err != nil {
		return embedding.Result{}, r.err
	}
	r.mu.Lock()
	r.requests = append(r.requests, req)
	r.mu.Unlock()

	vectors := make([]embedding.Vector, len(req.Inputs))
	for i := range req.Inputs {
		vectors[i] = embedding.Vector{float32(i), 1}
	}
	if r.shortByID {
		vectors = vectors[:len(vectors)-1]
	}
	return embedding.Result{Model: req.Model, Vectors: vectors}, nil
}

func TestBatchEmbedderFromEmbedder_EmptyBatch(t *testing.T) {
	t.Parallel()
	bridge := BatchEmbedderFromEmbedder(&recordingEmbedder{}, embedding.ModelRef{Name: "m"}, embedding.KindDocument)

	vectors, err := bridge.EmbedBatch(t.Context(), nil)
	require.NoError(t, err)
	assert.Nil(t, vectors)

	vectors, err = bridge.EmbedBatch(t.Context(), []string{})
	require.NoError(t, err)
	assert.Nil(t, vectors)
}

func TestBatchEmbedderFromEmbedder_OrderAndRequestPreserved(t *testing.T) {
	t.Parallel()
	fake := &recordingEmbedder{}
	model := embedding.ModelRef{Name: "nomic-embed"}
	bridge := BatchEmbedderFromEmbedder(fake, model, embedding.KindDocument)

	texts := []string{"alpha", "beta", "gamma"}
	vectors, err := bridge.EmbedBatch(t.Context(), texts)
	require.NoError(t, err)
	require.Len(t, vectors, len(texts))
	for i, v := range vectors {
		require.Len(t, v, 2)
		assert.Equal(t, float32(i), v[0], "vector order must match input order")
	}

	require.Len(t, fake.requests, 1)
	req := fake.requests[0]
	assert.Equal(t, model, req.Model)
	assert.Equal(t, embedding.KindDocument, req.Kind)
	assert.Equal(t, texts, req.Inputs)
}

func TestBatchEmbedderFromEmbedder_ErrorPropagation(t *testing.T) {
	t.Parallel()
	sentinel := errors.New("provider unavailable")
	bridge := BatchEmbedderFromEmbedder(&recordingEmbedder{err: sentinel}, embedding.ModelRef{}, embedding.KindQuery)

	vectors, err := bridge.EmbedBatch(t.Context(), []string{"x"})
	require.ErrorIs(t, err, sentinel)
	assert.Nil(t, vectors)
}

func TestBatchEmbedderFromEmbedder_RejectsMismatchedVectorCount(t *testing.T) {
	t.Parallel()
	bridge := BatchEmbedderFromEmbedder(&recordingEmbedder{shortByID: true}, embedding.ModelRef{}, embedding.KindDocument)

	vectors, err := bridge.EmbedBatch(t.Context(), []string{"a", "b"})
	require.ErrorIs(t, err, embedding.ErrInvalidResult)
	assert.Nil(t, vectors)
}

func TestBatchEmbedderFromEmbedder_NilEmbedder(t *testing.T) {
	t.Parallel()
	bridge := BatchEmbedderFromEmbedder(nil, embedding.ModelRef{}, embedding.KindDocument)

	_, err := bridge.EmbedBatch(t.Context(), []string{"a"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no underlying embedding.Embedder")
}

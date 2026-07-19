package retrieval

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary/embedding"
)

type embedFunc func(context.Context, embeddings.Request) (embeddings.Result, error)

func (f embedFunc) Embed(ctx context.Context, request embeddings.Request) (embeddings.Result, error) {
	return f(ctx, request)
}

func TestAttachEmbeddings(t *testing.T) {
	t.Parallel()

	results := []*Result{{ID: "a"}, {ID: "b"}}
	err := AttachEmbeddings(results, []embeddings.Vector{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatalf("AttachEmbeddings() error = %v", err)
	}
	if results[0].Embedding[0] != 1 || results[0].Embedding[1] != 2 || results[1].Embedding[0] != 3 || results[1].Embedding[1] != 4 {
		t.Fatalf("embeddings not attached: %#v", results)
	}
}

func TestAttachEmbeddingsCountMismatch(t *testing.T) {
	t.Parallel()

	err := AttachEmbeddings([]*Result{{ID: "a"}}, []embeddings.Vector{{1}, {2}})
	if err == nil {
		t.Fatal("AttachEmbeddings() error = nil, want mismatch error")
	}
	if !errors.Is(err, ErrEmbeddingCountMismatch) {
		t.Fatalf("AttachEmbeddings() error = %v, want ErrEmbeddingCountMismatch", err)
	}
}

func TestEmbedResults(t *testing.T) {
	t.Parallel()

	results := []*Result{
		{ID: "a", Content: "alpha"},
		{ID: "b", Content: "beta"},
	}
	stub := embedFunc(func(_ context.Context, request embeddings.Request) (embeddings.Result, error) {
		vectors := make([]embeddings.Vector, len(request.Inputs))
		for i := range request.Inputs {
			vectors[i] = embeddings.Vector{float32(i + 1), 42}
		}
		return embeddings.Result{Vectors: vectors}, nil
	})

	if err := EmbedResults(context.Background(), stub, results); err != nil {
		t.Fatalf("EmbedResults() error = %v", err)
	}
	if results[0].Embedding[0] != 1 || results[0].Embedding[1] != 42 || results[1].Embedding[0] != 2 || results[1].Embedding[1] != 42 {
		t.Fatalf("embeddings not attached: %#v", results)
	}
	if err := EmbedResults(context.Background(), stub, nil); err != nil {
		t.Fatalf("EmbedResults(nil) error = %v", err)
	}
}

func TestRerankEmbedding(t *testing.T) {
	t.Parallel()

	scorer := NewScorer(DefaultWeights())
	results := []*Result{
		{ID: "a", Content: "alpha", Embedding: []float64{1, 0}},
		{ID: "b", Content: "beta", Embedding: []float64{0, 1}},
	}

	ranked := scorer.RerankEmbedding(embeddings.Vector{1, 0}, "alpha", results)
	if ranked[0].ID != "a" {
		t.Fatalf("RerankEmbedding() top result = %q, want a", ranked[0].ID)
	}
}

func TestEmbeddingVectors(t *testing.T) {
	t.Parallel()

	vecs := []embeddings.Vector{{1.0, 2.0}, {3.0, 4.0}}
	got := EmbeddingVectors(vecs)
	if len(got) != 2 || got[0][0] != 1.0 || got[1][1] != 4.0 {
		t.Fatalf("EmbeddingVectors = %v", got)
	}

	results := []*Result{nil, {ID: "b"}}
	if err := AttachEmbeddings(results, vecs); err != nil {
		t.Fatalf("AttachEmbeddings with nil item failed: %v", err)
	}
}

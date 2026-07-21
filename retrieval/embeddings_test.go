package retrieval

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/dotcommander/reliquary/embedding"
)

type embedFunc func(context.Context, embedding.Request) (embedding.Result, error)

func (f embedFunc) Embed(ctx context.Context, request embedding.Request) (embedding.Result, error) {
	return f(ctx, request)
}

func TestAttachEmbeddings(t *testing.T) {
	t.Parallel()

	results := []*Result{{ID: "a"}, {ID: "b"}}
	err := AttachEmbeddings(results, []embedding.Vector{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatalf("AttachEmbeddings() error = %v", err)
	}
	if results[0].Embedding[0] != 1 || results[0].Embedding[1] != 2 || results[1].Embedding[0] != 3 || results[1].Embedding[1] != 4 {
		t.Fatalf("embeddings not attached: %#v", results)
	}
}

func TestAttachEmbeddingsCountMismatch(t *testing.T) {
	t.Parallel()

	err := AttachEmbeddings([]*Result{{ID: "a"}}, []embedding.Vector{{1}, {2}})
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
	stub := embedFunc(func(_ context.Context, request embedding.Request) (embedding.Result, error) {
		vectors := make([]embedding.Vector, len(request.Inputs))
		for i := range request.Inputs {
			vectors[i] = embedding.Vector{float32(i + 1), 42}
		}
		return embedding.Result{Vectors: vectors}, nil
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

func TestEmbedResultsRejectsNilBeforeEmbedding(t *testing.T) {
	t.Parallel()

	called := false
	stub := embedFunc(func(context.Context, embedding.Request) (embedding.Result, error) {
		called = true
		return embedding.Result{}, nil
	})
	results := []*Result{{ID: "a", Content: "alpha", Embedding: []float64{9}}, nil}
	if err := EmbedResults(context.Background(), stub, results); !errors.Is(err, ErrNilResult) {
		t.Fatalf("EmbedResults() error = %v, want ErrNilResult", err)
	}
	if called {
		t.Fatal("EmbedResults called the embedder for a nil result")
	}
	if !reflect.DeepEqual(results[0].Embedding, []float64{9}) {
		t.Fatalf("existing embedding mutated: %v", results[0].Embedding)
	}
}

func TestEmbedResultsRejectsMalformedBatchBeforeMutation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		result         embedding.Result
		wantCountError bool
	}{
		{
			name:           "count mismatch",
			result:         embedding.Result{Vectors: []embedding.Vector{{1, 0}}},
			wantCountError: true,
		},
		{
			name:   "ragged vectors",
			result: embedding.Result{Vectors: []embedding.Vector{{1, 0}, {1}}},
		},
		{
			name:   "non-finite vector",
			result: embedding.Result{Vectors: []embedding.Vector{{1, 0}, {float32(math.NaN()), 1}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			results := []*Result{
				{ID: "a", Content: "alpha", Embedding: []float64{9, 8}},
				{ID: "b", Content: "beta", Embedding: []float64{7, 6}},
			}
			before := [][]float64{append([]float64(nil), results[0].Embedding...), append([]float64(nil), results[1].Embedding...)}
			stub := embedFunc(func(context.Context, embedding.Request) (embedding.Result, error) {
				return tt.result, nil
			})
			err := EmbedResults(context.Background(), stub, results)
			if !errors.Is(err, embedding.ErrInvalidResult) {
				t.Fatalf("EmbedResults() error = %v, want ErrInvalidResult", err)
			}
			if errors.Is(err, ErrEmbeddingCountMismatch) != tt.wantCountError {
				t.Fatalf("errors.Is(ErrEmbeddingCountMismatch) = %v, want %v: %v", errors.Is(err, ErrEmbeddingCountMismatch), tt.wantCountError, err)
			}
			if !reflect.DeepEqual([][]float64{results[0].Embedding, results[1].Embedding}, before) {
				t.Fatalf("results mutated after invalid batch: %#v", results)
			}
		})
	}
}

func TestRerankEmbedding(t *testing.T) {
	t.Parallel()

	scorer := NewScorer(DefaultWeights())
	results := []*Result{
		{ID: "a", Content: "alpha", Embedding: []float64{1, 0}},
		{ID: "b", Content: "beta", Embedding: []float64{0, 1}},
	}

	ranked := scorer.RerankEmbedding(embedding.Vector{1, 0}, "alpha", results)
	if ranked[0].ID != "a" {
		t.Fatalf("RerankEmbedding() top result = %q, want a", ranked[0].ID)
	}
}

func TestEmbeddingVectors(t *testing.T) {
	t.Parallel()

	vecs := []embedding.Vector{{1.0, 2.0}, {3.0, 4.0}}
	got := EmbeddingVectors(vecs)
	if len(got) != 2 || got[0][0] != 1.0 || got[1][1] != 4.0 {
		t.Fatalf("EmbeddingVectors = %v", got)
	}

	results := []*Result{nil, {ID: "b"}}
	if err := AttachEmbeddings(results, vecs); err != nil {
		t.Fatalf("AttachEmbeddings with nil item failed: %v", err)
	}
}

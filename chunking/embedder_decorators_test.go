package chunking

import (
	"context"
	"errors"
	"testing"

	"github.com/dotcommander/reliquary/embedding"
)

type batchRecordingEmbedder struct {
	batches [][]string
	vectors int
}

func (e *batchRecordingEmbedder) Embed(_ context.Context, req embedding.Request) (embedding.Result, error) {
	e.batches = append(e.batches, req.Inputs)
	e.vectors += len(req.Inputs)
	out := make([]embedding.Vector, len(req.Inputs))
	for i := range out {
		out[i] = embedding.Vector{float32(i), 1}
	}
	return embedding.Result{Model: req.Model, Vectors: out}, nil
}

func TestLimitBatchSizeSplitsAndReassemblesInOrder(t *testing.T) {
	t.Parallel()
	inner := &batchRecordingEmbedder{}
	limited := LimitBatchSize(inner, 2)
	texts := []string{"a", "b", "c", "d", "e"}
	res, err := limited.Embed(context.Background(), embedding.Request{Inputs: texts})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(res.Vectors) != len(texts) {
		t.Fatalf("got %d vectors, want %d", len(res.Vectors), len(texts))
	}
	if inner.vectors != len(texts) {
		t.Fatalf("embedded %d inputs, want %d", inner.vectors, len(texts))
	}
	if len(inner.batches) != 3 { // 2 + 2 + 1
		t.Fatalf("got %d provider calls, want 3", len(inner.batches))
	}
}

func TestLimitBatchSizePassthroughWhenUnlimited(t *testing.T) {
	t.Parallel()
	inner := &batchRecordingEmbedder{}
	same := LimitBatchSize(inner, 0)
	if same != embedding.Embedder(inner) {
		t.Fatal("max <= 0 must return the embedder unchanged")
	}
	if _, err := same.Embed(context.Background(), embedding.Request{Inputs: []string{"a"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(inner.batches) != 1 {
		t.Fatalf("got %d calls, want 1", len(inner.batches))
	}
}

type failingEmbedder struct{}

func (failingEmbedder) Embed(context.Context, embedding.Request) (embedding.Result, error) {
	return embedding.Result{}, errors.New("provider down")
}

func TestLimitBatchSizePropagatesErrors(t *testing.T) {
	t.Parallel()
	limited := LimitBatchSize(failingEmbedder{}, 1)
	if _, err := limited.Embed(context.Background(), embedding.Request{Inputs: []string{"a", "b"}}); err == nil {
		t.Fatal("expected error propagation")
	}
}

type stubBatchEmbedder struct {
	calls int
	err   error
}

func (s *stubBatchEmbedder) EmbedBatch(context.Context, []string) ([][]float32, error) {
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	return [][]float32{{0, 1}}, nil
}

func TestLatchedBatchEmbedderRecordsFirstError(t *testing.T) {
	t.Parallel()
	first := errors.New("first failure")
	stub := &stubBatchEmbedder{err: first}
	latch := LatchBatchEmbedder(stub)
	if _, err := latch.EmbedBatch(context.Background(), []string{"a"}); err == nil {
		t.Fatal("EmbedBatch must fail through unchanged")
	}
	stub.err = errors.New("second failure")
	if _, err := latch.EmbedBatch(context.Background(), []string{"b"}); err == nil {
		t.Fatal("EmbedBatch must fail through unchanged")
	}
	if !errors.Is(latch.Err(), first) {
		t.Fatalf("Err() = %v, want the first error", latch.Err())
	}
	if stub.calls != 2 {
		t.Fatalf("got %d calls, want 2", stub.calls)
	}
}

func TestLatchedBatchEmbedderCleanRun(t *testing.T) {
	t.Parallel()
	latch := LatchBatchEmbedder(&stubBatchEmbedder{})
	if _, err := latch.EmbedBatch(context.Background(), []string{"a"}); err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if err := latch.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
}

func TestLatchBatchEmbedderNilInner(t *testing.T) {
	t.Parallel()
	if LatchBatchEmbedder(nil) != nil {
		t.Fatal("nil inner must return nil")
	}
}
